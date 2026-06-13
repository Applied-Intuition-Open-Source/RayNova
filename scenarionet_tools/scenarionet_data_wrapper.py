# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IterableDataset that reads ScenarioNet-formatted scenarios directly from a
local directory, bypassing the S3/Lance pipeline used by `NuPlanDataWrapper`.

Layout assumed (matches what `write_nuscenes_to_s3.py` ingests):

    <root>/
        dataset_mapping.pkl
        dataset_summary.pkl
        <subset>/
            *.pkl   # each one a pickled ScenarioDescription dict

The wrapper subclasses `NuPlanDataWrapper` and only swaps the I/O layer, so
all per-frame preprocessing (scenario meta, sequence sampling, bbox coord
conversion, map sampling, ego-motion transforms) is reused unchanged.
Output of `__iter__` is therefore byte-for-byte identical to
`NuPlanDataWrapper.__iter__`: a `List[frame_dict_after_pipeline]` of length
`data_config['horizon']`.
"""

import os
import os.path as osp
import pathlib
import pickle
import random

import numpy as np
import PIL
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from metadrive.scenario import ScenarioDescription

from nuscenes_tools.nuscenes_utils.pipelines import Compose  # noqa: F401
import nuscenes_tools.nuscenes_utils.pipelines as pipelines
from nuscenes_tools.nuscenes_utils.box3d_instance import LiDARInstance3DBoxes
import scenarionet_tools.map_utils as map_utils


_SUMMARY_FILES = {'dataset_summary.pkl', 'dataset_mapping.pkl'}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def convert_bbox(xyz, heading, rotation, translation):
    is_torch = torch.is_tensor(xyz)
    if is_torch:
        device = xyz.device
        xyz = xyz.numpy()
        heading = heading.numpy()
        rotation = rotation.numpy()
        translation = translation.numpy()
    xyz = xyz @ rotation.T + translation[None]
    rot = R.from_matrix(rotation[None]).as_euler('xyz', degrees=False)
    heading = heading + rot[..., -1]
    if is_torch:
        xyz = torch.from_numpy(xyz).to(device)
        heading = torch.from_numpy(heading).to(device)
    return xyz, heading


# ---------------------------------------------------------------------------
# Base dataset — all preprocessing logic, no I/O
# ---------------------------------------------------------------------------

class NuPlanDataWrapper(torch.utils.data.IterableDataset):
    def __init__(self, data_config, rank=0, world_size=1, pipeline=None, leaveout=False, randomize=True, load_image=2, seed=None, random_drop_cond=True):
        # load image:  0 (w/o image), 1 (first frame only), 2 (all images)
        self.data_config = data_config
        self.data_path = data_config['dataset_path']
        self.leaveout = leaveout
        self.load_image = load_image
        self.random_drop_cond = random_drop_cond

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self.rank = rank
        self.world_size = world_size
        self.dataset_name = data_config.get('dataset_name', 'nuplan')
        self.randomize = randomize
        self.seed = seed

        if self.dataset_name == 'nuscenes':
            self.ground_z = -1.84
        else:
            self.ground_z = -0.3186

    def _get_scenario_info(self, scenario):
        scenario_dict = {}
        if 'timeofday' in scenario['metadata']:
            date_time = scenario['metadata']['timeofday']
        else:
            date_time = scenario['metadata']['log_name'].split('_')[0]

        if self.dataset_name == 'nuscenes':
            scenario_dict['location'] = scenario['metadata']['map']
            date_time = scenario['metadata']['timeofday']
            timeofday = ':'.join(date_time.split('-')[3:5])
            scenario_dict['timeofday'] = timeofday
        else:
            log_name = scenario['metadata']['log_name']
            date_time = log_name.split('_')[0]
            timeofday = ':'.join(date_time.split('.')[3:5])
            scenario_dict['timeofday'] = timeofday
            scenario_dict['location'] = scenario['metadata']['map']

            if 'language_description' in scenario:
                scenario_dict['language_description'] = scenario['language_description']
            else:
                description = scenario['metadata']['scenario_extraction_info']['scenario_name']
                objects = scenario['metadata']['object_summary']
                sdc_id = scenario['metadata']['sdc_id']

                object_categories = []
                for obj_id in objects:
                    if sdc_id != obj_id:
                        obj_cat = objects[obj_id]['type'].lower()
                        if obj_cat not in object_categories:
                            object_categories.append(obj_cat)
                object_categories = ', '.join(object_categories)
                description = description + '. ' + object_categories
                scenario_dict['description'] = description

        return scenario_dict

    def _get_object_info(self, scenario, id):
        gt_bboxes_3d = []
        gt_names_3d = []

        sdc_id = scenario['metadata']['sdc_id']
        tracks = scenario['tracks']
        for obj_id in tracks:
            if obj_id == 'sdc_id':
                continue
            obj_info = tracks[obj_id]
            if not obj_info['state']['valid'][id]:
                continue
            position = obj_info['state']['position'][id]
            heading = obj_info['state']['heading'][id][None]
            velocity = obj_info['state']['velocity'][id]
            length = obj_info['state']['length'][id]
            width = obj_info['state']['width'][id]
            height = obj_info['state']['height'][id]

            obj_size = np.concatenate([length, width, height], axis=0)

            bbox = np.concatenate([position, obj_size, heading], axis=0)
            gt_bboxes_3d.append(bbox)

            label = obj_info['type']
            gt_names_3d.append(label)

        if len(gt_bboxes_3d) > 0:
            gt_bboxes_3d = np.stack(gt_bboxes_3d, axis=0)
        else:
            gt_bboxes_3d = np.zeros((0, 7))
        gt_names_3d = np.array(gt_names_3d)

        return {'gt_bboxes_3d': gt_bboxes_3d, 'gt_names_3d': gt_names_3d}

    def _get_camera_info(self, camera_sensors, id_in_seq):
        all_camera_info = {}

        for cam_name in camera_sensors:
            cam_data = camera_sensors[cam_name]
            cam_info = {}
            raw_path = cam_data['cam_abs_path']
            if raw_path.startswith('s3://'):
                # Remote path: normalise bucket name (underscore -> hyphen).
                cam_info['data_path'] = raw_path.replace('research_datasets', 'research-datasets')
            elif os.path.isabs(raw_path):
                # Already an absolute local path.
                cam_info['data_path'] = raw_path
            else:
                # Relative path: resolve against images_dir from data_config.
                images_dir = os.path.join(self.data_config['dataset_path'], 'sensor_blobs')
                cam_info['data_path'] = os.path.join(images_dir, raw_path)

            cam_info['cam_intrinsic'] = np.array(cam_data['cam_intrinsic'], dtype=np.float32)
            cam_info['sensor2lidar_translation'] = np.array(cam_data['sensor2lidar_translation'], dtype=np.float32)
            cam_info['sensor2lidar_rotation'] = np.array(cam_data['sensor2lidar_rotation'], dtype=np.float32)

            cam_info['ego2global_rotation'] = Quaternion(np.array(cam_data['ego2global_rotation'], dtype=np.float32)).rotation_matrix
            cam_info['ego2global_translation'] = np.array(cam_data['ego2global_translation'], dtype=np.float32)

            sensor2lidar = np.eye(4)
            sensor2lidar[:3, :3] = cam_info['sensor2lidar_rotation']
            sensor2lidar[:3, 3] = cam_info['sensor2lidar_translation']

            ego2global = np.eye(4)
            ego2global[:3, :3] = cam_info['ego2global_rotation']
            ego2global[:3, 3] = cam_info['ego2global_translation']

            sensor2ego = np.eye(4)
            sensor2ego[:3, :3] = Quaternion(np.array(cam_data['sensor2ego_rotation'], dtype=np.float32)).rotation_matrix
            sensor2ego[:3, 3] = np.array(cam_data['sensor2ego_translation'], dtype=np.float32)

            global2lidar = sensor2lidar @ np.linalg.inv(sensor2ego) @ np.linalg.inv(ego2global)
            cam_info['global2lidar_rotation'] = global2lidar[:3, :3]
            cam_info['global2lidar_translation'] = global2lidar[:3, 3]

            if 'cam_distortion' in cam_data:
                cam_info['distortion'] = cam_data['cam_distortion']

            if self.load_image == 0 or (self.load_image == 1 and id_in_seq >= 3):
                cam_info['load_image'] = False
            else:
                cam_info['load_image'] = True

            all_camera_info[cam_name] = cam_info

        return all_camera_info

    def _get_sensor_sequence_info(self, scenario):
        horizon = self.data_config['horizon']
        if horizon == 'all':
            assert not self.data_config['random_interval']
            horizon = len(scenario['raw_sensors']) // self.data_config['interval']
        intervals = []
        if self.data_config['random_interval']:
            intervals = [random.randint(0, self.data_config['interval']) for _ in range(horizon-1)]
        else:
            intervals = [self.data_config['interval']] * (horizon - 1)

        intervals = [0] + intervals

        raw_sensors = scenario['raw_sensors']
        total_len = sum(intervals)
        sids = list(range(len(raw_sensors)-total_len))
        sid = random.choice(sids)
        if 'first_frame' in self.data_config:
            sid = self.data_config['first_frame']

        timesteps = np.cumsum(intervals)
        seq_ids = timesteps + sid
        assert len(seq_ids) == horizon and seq_ids[-1] < len(raw_sensors)

        image_dict_sequence = []
        bbox_sequence = []
        for id_in_seq, id in enumerate(seq_ids):
            all_camera_info = self._get_camera_info(raw_sensors[id]['images'], id_in_seq)
            if self.dataset_name == 'nuscenes':
                all_camera_info['description'] = raw_sensors[id]['metadata']['description']
            image_dict_sequence.append(all_camera_info)

            if 'load_bbox' in self.data_config and self.data_config['load_bbox']:
                all_object_info = self._get_object_info(scenario, id)
                bbox_sequence.append(all_object_info)

        if horizon > 1:
            if self.dataset_name == 'nuscenes':
                timesteps = timesteps / ((horizon - 1) * 2)
            else:
                timesteps = timesteps / ((horizon - 1) * 10)
        else:
            timesteps = np.zeros_like(timesteps)
        return image_dict_sequence, bbox_sequence, timesteps, seq_ids

    def _filter_bbox(self, xyz):
        obj_range = self.data_config['object_range']
        flag = (xyz[:, 0] > obj_range[0]) & (xyz[:, 1] > obj_range[1]) & (xyz[:, 2] > obj_range[2]) \
            & (xyz[:, 0] < obj_range[3]) & (xyz[:, 1] < obj_range[4]) & (xyz[:, 2] < obj_range[5])
        return flag

    def _convert_and_filter_bbox_coordinate(self, bbox_sequence, image_dict_sequence):
        if self.dataset_name == 'nuscenes':
            cam_name = 'CAM_FRONT'
        else:
            cam_name = 'CAM_F0'
        first_frame_info = image_dict_sequence[0]
        first_frame_global2lidar_rotation = first_frame_info[cam_name]['global2lidar_rotation']
        first_frame_global2lidar_translation = first_frame_info[cam_name]['global2lidar_translation']

        for frame_box, frame_info in zip(bbox_sequence, image_dict_sequence):
            if len(frame_box['gt_bboxes_3d']) == 0:
                continue
            curr_frame_global2lidar_rotation = frame_info[cam_name]['global2lidar_rotation']
            curr_frame_global2lidar_translation = frame_info[cam_name]['global2lidar_translation']

            xyz = frame_box['gt_bboxes_3d'][:, :3]
            heading = frame_box['gt_bboxes_3d'][:, 6]

            curr_xyz, _ = convert_bbox(xyz, heading, curr_frame_global2lidar_rotation, curr_frame_global2lidar_translation)
            flag = self._filter_bbox(curr_xyz)
            frame_box['gt_bboxes_3d'] = frame_box['gt_bboxes_3d'][flag]
            frame_box['gt_names_3d'] = frame_box['gt_names_3d'][flag]

            if len(frame_box['gt_bboxes_3d']) == 0:
                continue
            new_xyz, new_heading = convert_bbox(frame_box['gt_bboxes_3d'][:, :3], frame_box['gt_bboxes_3d'][:, 6], first_frame_global2lidar_rotation, first_frame_global2lidar_translation)
            frame_box['gt_bboxes_3d'][:, :3] = new_xyz
            frame_box['gt_bboxes_3d'][:, 6] = new_heading

            frame_box['gt_bboxes_3d'] = LiDARInstance3DBoxes(frame_box['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5))

        return bbox_sequence

    def _get_map_info(self, scenario, image_dict_sequence):
        map_features = scenario['map_features']
        if self.dataset_name == 'nuscenes':
            cam_name = 'CAM_FRONT'
        else:
            cam_name = 'CAM_F0'
        first_frame_info = image_dict_sequence[0]
        first_frame_global2lidar_rotation = first_frame_info[cam_name]['global2lidar_rotation']
        first_frame_global2lidar_translation = first_frame_info[cam_name]['global2lidar_translation']

        map_sequence = []

        for frame_id, frame_info in enumerate(image_dict_sequence):
            frame_map_type = []
            frame_map_points = []
            if len(map_features) > 0:
                curr_frame_global2lidar_rotation = frame_info[cam_name]['global2lidar_rotation']
                curr_frame_global2lidar_translation = frame_info[cam_name]['global2lidar_translation']

                curr_to_first_rotation = first_frame_global2lidar_rotation @ curr_frame_global2lidar_rotation.T
                curr_to_first_translation = -first_frame_global2lidar_rotation @ curr_frame_global2lidar_rotation.T @ curr_frame_global2lidar_translation + first_frame_global2lidar_translation

                if self.dataset_name == 'nuscenes':
                    default_z = 0
                else:
                    default_z = -curr_frame_global2lidar_translation[-1] + self.ground_z

                for map_key in map_features:
                    if 'polyline' in map_features[map_key]:
                        polyline = map_utils.convert_points(map_features[map_key]['polyline'][..., :2], curr_frame_global2lidar_rotation, curr_frame_global2lidar_translation, ground_z=self.ground_z, default_z=default_z)
                    else:
                        polyline = None
                    if 'polygon' in map_features[map_key]:
                        polygon = map_utils.convert_points(map_features[map_key]['polygon'][..., :2], curr_frame_global2lidar_rotation, curr_frame_global2lidar_translation, ground_z=self.ground_z, default_z=default_z)
                    else:
                        polygon = None

                    if polyline is None:
                        flag = map_utils.filter_map_elements(polygon)
                    else:
                        flag = map_utils.filter_map_elements(polyline)

                    if not flag:
                        continue

                    if polyline is None:
                        plane_coeffs = map_utils.fit_plane_from_points(polygon)
                        segments = map_utils.split_polygon_max_area(polygon, 20)
                    else:
                        segments = map_utils.resample_polyline(polyline, 10)

                    for segment in segments:
                        if polyline is None:
                            points = map_utils.sample_points_from_polygon(segment, self.data_config['map_sample_points_num'])
                            points_z = map_utils.compute_z(plane_coeffs, points)
                            points = np.concatenate([points[:, :2], points_z[:, None]], axis=1)
                        else:
                            points = map_utils.sample_points_from_polyline(segment, self.data_config['map_sample_points_num'])

                        points_dist = np.linalg.norm(points[:, :2], axis=1)
                        points_dist = points_dist.max()
                        if points_dist > 50:
                            continue

                        assert points.shape[-1] == 3
                        points = map_utils.convert_points(points, curr_to_first_rotation, curr_to_first_translation)

                        frame_map_type.append(map_features[map_key]['type'].lower())
                        frame_map_points.append(points)

            if len(frame_map_points) == 0:
                frame_map_points = np.zeros((0, self.data_config['map_sample_points_num'], 3))
            else:
                frame_map_points = np.stack(frame_map_points, axis=0)
            frame_map_info = {
                'map_type_names': frame_map_type,
                'map_sampled_points': frame_map_points,
            }
            map_sequence.append(frame_map_info)

        return map_sequence

    def _preprocess_sample(self, item):
        scenario = pickle.loads(item['scenario'])
        scenario = ScenarioDescription(scenario)

        data_info = {}
        scenario_dict = self._get_scenario_info(scenario)
        data_info.update(scenario_dict)
        data_info['sample_idx'] = item['scenario_id']

        image_dict_sequence, bbox_sequence, timesteps, seq_ids = self._get_sensor_sequence_info(scenario)
        for camera_infos in image_dict_sequence:
            if len(camera_infos) < self.data_config['Ncams']:
                return None

        if 'load_bbox' in self.data_config and self.data_config['load_bbox']:
            bbox_sequence = self._convert_and_filter_bbox_coordinate(bbox_sequence, image_dict_sequence)

        if 'load_map' in self.data_config and self.data_config['load_map']:
            map_sequence = self._get_map_info(scenario, image_dict_sequence)

        frame_dict_sequence = []
        for fid, camera_infos in enumerate(image_dict_sequence):
            frame_dict = {}
            frame_data_info = data_info.copy()
            if self.dataset_name == 'nuscenes':
                sentences = camera_infos['description'].split('.')
                if sentences[-1] == '':
                    sentences = sentences[:-1]
                if self.random_drop_cond:
                    sentences = [item for item in sentences if random.random() > 0.25]
                description = '.'.join(sentences) + '.'
                if description == '.':
                    description = ''
                frame_data_info['description'] = description
                del camera_infos['description']
            else:
                if 'language_description' in frame_data_info:
                    try:
                        description = frame_data_info['language_description'][seq_ids[fid]]['description']
                        sentences = description.split('.')
                        if sentences[-1] == '':
                            sentences = sentences[:-1]
                        if self.random_drop_cond:
                            sentences = [item for item in sentences if random.random() > 0.25]
                        description = '.'.join(sentences) + '.'
                        if description == '.':
                            description = ''
                        frame_data_info['description'] = description
                    except:
                        print(item['scenario_id'])
                        print(frame_data_info['language_description'][seq_ids[fid]])
                        frame_data_info['description'] = ''

            frame_dict.update(frame_data_info)
            frame_dict['img_info'] = camera_infos
            frame_dict['img_path'] = {key: camera_infos[key]['data_path'] for key in camera_infos}
            frame_dict['timestep'] = timesteps[fid]

            if 'load_bbox' in self.data_config and self.data_config['load_bbox']:
                frame_dict.update(bbox_sequence[fid])

            if 'load_map' in self.data_config and self.data_config['load_map']:
                frame_dict.update(map_sequence[fid])

            frame_dict_sequence.append(frame_dict)

        frame_dict_sequence = self._preprocess_motion(frame_dict_sequence)

        return frame_dict_sequence

    def _preprocess_motion(self, frame_dict_sequence):
        if self.dataset_name == 'nuscenes':
            cam_name = 'CAM_FRONT'
        else:
            cam_name = 'CAM_F0'
        for id, frame_dict in enumerate(frame_dict_sequence):
            if id == 0:
                curr_to_prev_lidar_rt = np.eye(4)
                curr_to_first_lidar_rt = np.eye(4)
            else:
                cur_img_info = frame_dict['img_info']
                cur_global2lidar_rotation = cur_img_info[cam_name]['global2lidar_rotation']
                cur_global2lidar_translation = cur_img_info[cam_name]['global2lidar_translation']
                cur_global2lidar = np.eye(4)
                cur_global2lidar[:3, :3] = cur_global2lidar_rotation
                cur_global2lidar[:3, 3] = cur_global2lidar_translation

                prev_img_info = frame_dict_sequence[id-1]['img_info']
                prev_global2lidar_rotation = prev_img_info[cam_name]['global2lidar_rotation']
                prev_global2lidar_translation = prev_img_info[cam_name]['global2lidar_translation']
                prev_global2lidar = np.eye(4)
                prev_global2lidar[:3, :3] = prev_global2lidar_rotation
                prev_global2lidar[:3, 3] = prev_global2lidar_translation

                first_img_info = frame_dict_sequence[0]['img_info']
                first_global2lidar_rotation = first_img_info[cam_name]['global2lidar_rotation']
                first_global2lidar_translation = first_img_info[cam_name]['global2lidar_translation']
                first_global2lidar = np.eye(4)
                first_global2lidar[:3, :3] = first_global2lidar_rotation
                first_global2lidar[:3, 3] = first_global2lidar_translation

                curr_to_prev_lidar_rt = prev_global2lidar @ np.linalg.inv(cur_global2lidar)
                curr_to_first_lidar_rt = first_global2lidar @ np.linalg.inv(cur_global2lidar)

            frame_dict['curr_to_prev_lidar_rt'] = torch.from_numpy(curr_to_prev_lidar_rt).float()
            frame_dict['curr_to_first_lidar_rt'] = torch.from_numpy(curr_to_first_lidar_rt).float()
        return frame_dict_sequence


# ---------------------------------------------------------------------------
# ScenarioNet local-file I/O
# ---------------------------------------------------------------------------

def _list_scenario_files(root: pathlib.Path):
    """Walk a ScenarioNet root and return a sorted list of scenario pkl paths.

    Mirrors the loop in `write_nuscenes_to_s3.load_all_scenarios`: any `*.pkl`
    file is accepted as long as it isn't one of the dataset-level summary
    files. Sub-directories are walked one level deep (matching the
    `<root>/<subset>/*.pkl` layout produced by the ScenarioNet converter).
    """
    files = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            for pkl in sorted(entry.glob('*.pkl')):
                if pkl.name in _SUMMARY_FILES:
                    continue
                files.append(pkl)
        elif entry.is_file() and entry.suffix == '.pkl' and entry.name not in _SUMMARY_FILES:
            files.append(entry)
    return files


def build_scenarionet_dataset(args, rank, world_size, source_id=None,
                              dataset_name='nuscenes'):
    """Mirrors `nuplan_data_wrapper.build_nuscenes_dataset` but drives a
    `ScenarioNetDataWrapper` that reads pkl files from `args.data_path`."""
    dataset_path = args.data_path
    if source_id is not None:
        dataset_path = dataset_path.split('+')[source_id]

    if args.pn == '0.25M':
        input_size = (384, 672)
    elif args.pn == '0.06M':
        input_size = (192, 336)
    else:
        assert args.pn == '1M'
        input_size = (768, 1344)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if dataset_name == 'nuscenes':
        cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        ncams = 6
        src_size = (900, 1600)
        # nuscenes keyframes are 2 Hz; nuplan ticks are 10 Hz.
        sample_interval = args.sample_interval // 5
    elif dataset_name == 'nuplan':
        cams = ['CAM_F0', 'CAM_R0', 'CAM_R1', 'CAM_R2',
                'CAM_B0', 'CAM_L2', 'CAM_L1', 'CAM_L0']
        ncams = 8
        src_size = (1080, 1920)
        sample_interval = args.sample_interval
    else:
        raise NotImplementedError(f'unknown dataset_name={dataset_name}')

    data_config = {
        'dataset_name': dataset_name,
        'dataset_path': dataset_path,
        'horizon': args.max_horizon,
        'interval': sample_interval,
        'random_interval': args.random_sample_interval,

        'load_bbox': args.object_condition,
        'object_range': (-50, -50, -10, 50, 50, 10),

        'cams': cams,
        'Ncams': ncams,
        'input_size': input_size,
        'src_size': src_size,
        'keep_ratio': False,
        'mean': mean,
        'std': std,

        # Augmentation
        'resize': (0, 0),
        'rot': (0, 0),
        'flip': False,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.0,

        'load_map': args.map_condition,
        'map_sample_points_num': args.map_sample_points_num,
        'images_dir': getattr(args, 'images_dir', ''),
    }

    load_keys = ['img_inputs']
    if args.object_condition:
        load_keys.extend(['gt_bboxes_3d', 'gt_labels_3d'])
    if args.map_condition:
        load_keys.extend(['map_type_labels', 'map_sampled_points'])
        map_names = args.map_names
    else:
        map_names = None

    if args.object_condition:
        train_pipeline = [
            pipelines.LoadMultiViewImageFromFiles_BEVDet(data_config, is_train=True),
            pipelines.DefaultFormatBundle3D(
                class_names=args.class_names, map_names=map_names,
                with_gt=True, with_label=True),
            pipelines.Collect3D(keys=load_keys),
        ]
    else:
        train_pipeline = [
            pipelines.LoadMultiViewImageFromFiles_BEVDet(data_config, is_train=True),
            pipelines.DefaultFormatBundle3D(
                class_names=None, map_names=map_names,
                with_gt=False, with_label=False),
            pipelines.Collect3D(keys=load_keys),
        ]

    return ScenarioNetDataWrapper(data_config, rank, world_size, train_pipeline)


class ScenarioNetDataWrapper(NuPlanDataWrapper):
    """Read ScenarioNet pkl files from a local directory and emit the same
    frame-dict sequences as `NuPlanDataWrapper`."""

    def __init__(self, data_config, rank=0, world_size=1, pipeline=None,
                 leaveout=False, randomize=True, load_image=2, seed=None,
                 random_drop_cond=True):
        super().__init__(data_config, rank=rank, world_size=world_size,
                         pipeline=pipeline, leaveout=leaveout,
                         randomize=randomize, load_image=load_image,
                         seed=seed, random_drop_cond=random_drop_cond)
        # Eagerly index the dataset directory so all workers/ranks share the
        # same (sorted, deterministic) file list.
        root = pathlib.Path(self.data_path)
        if not root.exists():
            raise FileNotFoundError(f'ScenarioNet root does not exist: {root}')
        self.scenario_files = _list_scenario_files(root)
        if not self.scenario_files:
            raise RuntimeError(
                f'No scenario pkl files found under {root}; expected the '
                f'ScenarioNet `<root>/<subset>/*.pkl` layout.')

    # ------------------------------------------------------------------
    # I/O — replaces Lance fetch; everything downstream is unchanged.
    # ------------------------------------------------------------------
    def _load_scenario_item(self, pkl_path):
        """Return an `item`-shaped dict that `_preprocess_sample` can consume.

        The pkl on disk is exactly `pickle.dump(scenario_dict, ...)` output, so
        its raw bytes can be handed straight to `_preprocess_sample`, which
        does `pickle.loads(item['scenario'])` then `ScenarioDescription(...)`.
        """
        with open(pkl_path, 'rb') as f:
            raw = f.read()
        # Cheaply peek at the pickled dict to recover the scenario id without
        # paying a second deserialization on the hot path. _preprocess_sample
        # will deserialize for real.
        try:
            scenario_dict = pickle.loads(raw)
            scenario_id = scenario_dict.get('id') or pkl_path.stem
        except Exception:
            scenario_id = pkl_path.stem
        return {'scenario_id': scenario_id, 'scenario': raw}

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        num_workers = max(num_workers, 1)
        shard_rank = self.rank * num_workers + worker_id
        total_shards = self.world_size * num_workers

        # Static shard: stride across ranks * workers so each shard sees a
        # disjoint, deterministic subset of the dataset.
        shard_files = self.scenario_files[shard_rank::total_shards]
        if not shard_files:
            # Fallback: very small datasets — replicate a single file.
            shard_files = self.scenario_files[shard_rank % len(self.scenario_files):
                                              shard_rank % len(self.scenario_files) + 1]

        if self.randomize:
            base_seed = self.seed if self.seed is not None else random.randint(0, 1 << 30)
            rng = random.Random(base_seed + shard_rank)
        else:
            rng = random.Random(shard_rank)

        epoch = 0
        while True:
            order = list(shard_files)
            if self.randomize:
                rng.shuffle(order)
            for pkl_path in order:
                try:
                    item = self._load_scenario_item(pkl_path)
                except (OSError, pickle.UnpicklingError) as e:
                    print(f'invalid scenario pkl {pkl_path}: {e}', flush=True)
                    continue

                sequence_data = self._preprocess_sample(item)
                if sequence_data is None:
                    continue

                try:
                    if self.pipeline is not None:
                        sequence_data = [self.pipeline(frame) for frame in sequence_data]
                except (PIL.UnidentifiedImageError, OSError, FileNotFoundError):
                    print(f'invalid data: {item["scenario_id"]}', flush=True)
                    continue

                yield sequence_data
                del sequence_data

            epoch += 1
