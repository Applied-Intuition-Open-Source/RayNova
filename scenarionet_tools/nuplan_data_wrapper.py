import torch
import lance
import os
import random
import pickle
import copy
import PIL
import time
import copy
import lance
import lance.torch.data
from typing import TypedDict
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from lance.sampler import ShardedFragmentSampler, ShardedBatchSampler
from functools import partial
import json
from metadrive.scenario import ScenarioDescription
import botocore.exceptions

from nuscenes_tools.nuscenes_utils.pipelines import Compose
import nuscenes_tools.nuscenes_utils.pipelines as pipelines
from nuscenes_tools.nuscenes_utils.box3d_instance import LiDARInstance3DBoxes
import scenarionet_tools.map_utils as map_utils

import scenarionet_tools.storage as storage
from infinity.utils.s3_file_utils import load_bytes_file, download_s3_folder

from shapely.geometry import Polygon, Point
from shapely.ops import triangulate

import logging

logging.getLogger("botocore").setLevel(logging.CRITICAL)
# import sys
# sys.path.append('..')
# import infinity.utils.dist as dist


os.environ['AWS_ACCESS_KEY_ID'] = 'cd0146b0fd5c24625a928b242d19f7e0dec18424'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'HN7mDT0pooo+3E40lyab8rrNfIied/33pCbpyrSEDuA='
os.environ['AWS_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com'
os.environ['AWS_DEFAULT_REGION'] = 'us-phoenix-1'
# os.environ['AWS_DEFAULT_REGION'] = 'us-chicago-1'
# os.environ['AWS_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com'

os.environ['PREAUTH_URL'] = 'https://idskhu5vqvtl.objectstorage.us-phoenix-1.oci.customer-oci.com/p/ofkGTeRQaWyr0mNvkheVidOQYGEjr4OmEhEAi3EECl_UjuMeqtvu8mKr-k22ixWw/n/idskhu5vqvtl/b/research_datasets/o/remote_deps'




def _invert_SE3(transforms: np.array) -> np.array:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape == (4, 4)
    Rinv = transforms[:3, :3].T
    out = np.zeros_like(transforms)
    out[:3, :3] = Rinv
    out[:3, 3] = -Rinv @ transforms[:3, 3]
    out[3, 3] = 1.0
    return out



def convert_bbox(xyz, heading, rotation, translation):
    is_torch = torch.is_tensor(xyz)
    if is_torch:
        device = xyz.device
        xyz = xyz.numpy()
        heading = heading.numpy()
        rotation = rotation.numpy()
        translation = translation.numpy()
    xyz = xyz @ rotation.T + translation[None]
    # heading = np.stack([np.zeros_like(heading), np.zeros_like(heading), heading], axis=1)
    rot = R.from_matrix(rotation[None]).as_euler('xyz', degrees=False)
    
    heading = heading + rot[..., -1]
    
    # heading = R.from_euler('xyz', heading, degrees=False).as_matrix()
    # heading = rotation[None] @ heading
    # heading = R.from_matrix(heading).as_euler('xyz', degrees=False)
    # heading = heading[..., -1]
    
    if is_torch:
        xyz = torch.from_numpy(xyz).to(device)
        heading = torch.from_numpy(heading).to(device)
    return xyz, heading


def _fetch_next(data_iter, data_path, rank, world_size, sampler):
    if 'chicago' in data_path:
        storage_options = {
            "aws_region": "us-chicago-1",
            "endpoint_url": "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com",
        }
    else:
        storage_options = {
            "aws_region": "us-phoenix-1",
            "endpoint_url": "https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com",
        }
    if data_iter == None:
        dataset = lance.dataset(
            data_path,
            storage_options=storage_options
        )
        
        lance_dataset = lance.torch.data.LanceDataset(
            # data_path,
            dataset,
            columns=["scenario_id", "scenario"],
            filter="raw_sensors_valid = true",
            batch_size=1,
            # batch_readahead=8,  # Control multi-threading reads.
            to_tensor_fn=to_tensor,
            sampler = sampler(
            # sampler = ShardedFragmentSampler(
            # sampler = ShardedBatchSampler(
                rank=rank,  # Rank of the current dataloader thread
                world_size=world_size,  # Total number of processes
                seed=random.randint(0, 100000),
            ),
        )
        data_iter = iter(lance_dataset)
    while True:
        try:
            output = next(data_iter)
            break
        except StopIteration:
            dataset = lance.dataset(
                data_path,
                storage_options=storage_options
            )
            
            lance_dataset = lance.torch.data.LanceDataset(
                # data_path,
                dataset,
                columns=["scenario_id", "scenario"],
                filter="raw_sensors_valid = true",
                batch_size=1,
                # batch_readahead=8,  # Control multi-threading reads.
                to_tensor_fn=to_tensor,
                sampler = sampler(
                # sampler = ShardedFragmentSampler(
                # sampler = ShardedBatchSampler(
                    rank=rank,  # Rank of the current dataloader thread
                    world_size=world_size,  # Total number of processes
                    seed=random.randint(0, 100000),
                ),
                # dataset_options=dataset_options
            )
            data_iter = iter(lance_dataset)
    return output, data_iter



def get_nextitem(data_iter, data_path, rank, world_size, sampler):
    max_retries = 10
    for attempt in range(max_retries):
        try:
            output, data_iter = _fetch_next(data_iter, data_path, rank, world_size, sampler)
            break  # Success, exit the loop
        except Exception as e:
            if "429 Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    sleep_time = (2**attempt) + random.uniform(0, 1)
                    print(
                        f"WARNING: Received 429 Too Many Requests, retrying in {sleep_time:.2f} seconds...", flush=True
                    )
                    time.sleep(sleep_time)
                else:
                    print(
                        "ERROR: Max retries reached for loading dataset from S3.", flush=True
                    )
                    raise  # Re-raise the error after max retries
            else:
                sleep_time = (2**attempt) + random.uniform(0, 1)
                print(e, flush=True)
                time.sleep(sleep_time)
                # raise  # Re-raise if not a 429 error
        if attempt == max_retries - 1:
            output, data_iter = _fetch_next(data_iter, data_path, rank, world_size, sampler)
    return output, data_iter


def to_tensor(batch, **kwargs):
    return batch


def build_nuscenes_dataset(args, rank, world_size, source_id=None):
    dataset_path = args.data_path
    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]

    if args.pn == '0.25M':
        input_size = (384, 672)
    elif args.pn == '0.06M':
        input_size = (192, 336)
    else:
        assert args.pn == '1M'
        input_size = (768, 1344)

        
    if source_id is not None:
        dataset_path = dataset_path.split('+')[source_id]
    
    data_config={
        'dataset_name': 'nuscenes',
        'dataset_path': dataset_path,
        'horizon': args.max_horizon,
        'interval': args.sample_interval // 5,
        'random_interval': args.random_sample_interval,
            
        'load_bbox': args.object_condition,
        'object_range': (-50, -50, -10, 50, 50, 10),

        'cams': ['CAM_FRONT', 'CAM_FRONT_RIGHT',  'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'],
        'Ncams': 6,
        #TODO: support higher resolution and bigger model
        'input_size': input_size,
        'src_size': (900, 1600),
        'keep_ratio': False,
        'mean': mean,
        'std': std,

        # Augmentation
        'resize': (0, 0),
        'rot': (0, 0),
        'flip': False,
        'crop_h': (0.0, 0.0),
        'resize_test':0.0,
        
        'load_map': args.map_condition,
        'map_sample_points_num': args.map_sample_points_num,
    }

    # lance_dataset = lance.dataset(
    #     dataset_path, storage_options=storage.make_s3_storage_options_from_env()
    # )

    # build augmentations
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
            pipelines.DefaultFormatBundle3D(class_names=args.class_names, map_names=map_names, with_gt=True, with_label=True),
            pipelines.Collect3D(keys=load_keys),
        ]
    else:
        train_pipeline = [
            pipelines.LoadMultiViewImageFromFiles_BEVDet(data_config, is_train=True),
            pipelines.DefaultFormatBundle3D(class_names=None, map_names=map_names, with_gt=False, with_label=False),
            pipelines.Collect3D(keys=load_keys),
        ]   

    # build dataset
    nuscenes_dataset = NuPlanDataWrapper(data_config, rank, world_size, train_pipeline)
    
    return nuscenes_dataset


def build_nuplan_dataset(args, rank, world_size, source_id=None):
    dataset_path = args.data_path
    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]
    
    if args.pn == '0.25M':
        input_size = (384, 672)
    elif args.pn == '0.06M':
        input_size = (192, 336)
    else:
        assert args.pn == '1M'
        input_size = (768, 1344)

    if source_id is not None:
        dataset_path = dataset_path.split('+')[source_id]

    data_config={
        'dataset_name': 'nuplan',
        'dataset_path': dataset_path,
        'horizon': args.max_horizon,
        'interval': args.sample_interval,
        'random_interval': args.random_sample_interval,
            
        'load_bbox': args.object_condition,
        'object_range': (-50, -50, -10, 50, 50, 10),

        'cams': ['CAM_F0', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0', 'CAM_L2', 'CAM_L1', 'CAM_L0'],
        'Ncams': 8,
        #TODO: support higher resolution and bigger model
        'input_size': input_size,
        'src_size': (1080, 1920),
        'keep_ratio': False,
        'mean': mean,
        'std': std,

        # Augmentation
        'resize': (0, 0),
        'rot': (0, 0),
        'flip': False,
        'crop_h': (0.0, 0.0),
        'resize_test':0.0,

        'load_map': args.map_condition,
        'map_sample_points_num': args.map_sample_points_num,
    }
    
    # lance_dataset = lance.dataset(
    #     dataset_path, storage_options=storage.make_s3_storage_options_from_env()
    # )

    # build augmentations
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
            pipelines.DefaultFormatBundle3D(class_names=args.class_names, map_names=map_names, with_gt=True, with_label=True),
            pipelines.Collect3D(keys=load_keys),
        ]
    else:
        train_pipeline = [
            pipelines.LoadMultiViewImageFromFiles_BEVDet(data_config, is_train=True),
            pipelines.DefaultFormatBundle3D(class_names=None, map_names=map_names, with_gt=False, with_label=False),
            pipelines.Collect3D(keys=load_keys),
        ]   

    # build dataset
    nuplan_dataset = NuPlanDataWrapper(data_config, rank, world_size, train_pipeline)
    
    return nuplan_dataset


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
            cam_info['data_path'] = cam_data['cam_abs_path'] # cam_abs_path in s3
            cam_info['data_path'] = cam_info['data_path'].replace('research_datasets', 'research-datasets')

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
            # if self.dataset_name != 'nuscenes' and len(map_features) > 0:
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
                        points = map_utils.convert_points(points, curr_to_first_rotation,  curr_to_first_translation)
                        
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
        
        # count = 0
        # total = 0
        # for map_key in scenario['map_features']:
        #     max_value = 0
        #     if 'polyline' in scenario['map_features'][map_key]:
        #         dist = np.linalg.norm(scenario['map_features'][map_key]['polyline'], axis=1)
        #         max_value = max(max_value, dist.min())
        #     if 'polygon' in scenario['map_features'][map_key]:
        #         dist = np.linalg.norm(scenario['map_features'][map_key]['polygon'], axis=1)
        #         max_value = max(max_value, dist.min())

        #     if max_value < 80:
        #         count += 1
        #     total += 1
        # print("in range", count, total)

        # print('\n\n\n')
        
        data_info = {}
        # data_info.update(scenario)
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
                # cur_img_info = frame_dict['img_info']
                # cur_ego2global_rotation = cur_img_info['CAM_F0']['ego2global_rotation']
                # cur_ego2global_translation = cur_img_info['CAM_F0']['ego2global_translation']
                # cur_ego2global = np.eye(4)
                # cur_ego2global[:3, :3] = cur_ego2global_rotation
                # cur_ego2global[:3, 3] = cur_ego2global_translation

                # prev_img_info = frame_dict_sequence[id-1]['img_info']
                # prev_ego2global_rotation = prev_img_info['CAM_F0']['ego2global_rotation']
                # prev_ego2global_translation = prev_img_info['CAM_F0']['ego2global_translation']
                # prev_ego2global = np.eye(4)
                # prev_ego2global[:3, :3] = prev_ego2global_rotation
                # prev_ego2global[:3, 3] = prev_ego2global_translation

                # _curr_to_prev_lidar_rt = np.linalg.inv(prev_ego2global) @ cur_ego2global

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
            frame_dict['curr_to_first_lidar_rt' ]= torch.from_numpy(curr_to_first_lidar_rt).float()
        return frame_dict_sequence
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        num_workers = max(num_workers, 1)

        # lance_dataset = lance.torch.data.LanceDataset(
        #     self.data_config['dataset_path'],
        #     columns=["scenario_id", "scenario"],
        #     filter="raw_sensors_valid = true",
        #     batch_size=1,
        #     # batch_readahead=8,  # Control multi-threading reads.
        #     to_tensor_fn=to_tensor,
        #     sampler = ShardedFragmentSampler(
        #         rank=self.rank*num_workers+worker_id,  # Rank of the current dataloader thread
        #         world_size=self.world_size*num_workers,  # Total number of processes
        #         randomize=True,
        #     )
        # )
        lance_iterator = None

        if self.leaveout:
            world_size = self.world_size*num_workers+1
        else:
            world_size = self.world_size*num_workers
        
        if self.dataset_name == 'nuscenes':
            sampler = partial(ShardedBatchSampler, randomize=self.randomize, seed=self.seed)
        else:
            sampler = partial(ShardedFragmentSampler, randomize=self.randomize, seed=self.seed)
        
        while True:
            sample, lance_iterator = get_nextitem(lance_iterator, self.data_config['dataset_path'], rank=self.rank*num_workers+worker_id, world_size=world_size, sampler=sampler)
            
            [sample] = sample.to_pylist()
                
            sequence_data = self._preprocess_sample(sample)
            if sequence_data is None:
                # print(f"sequence_data is {sequence_data}", flush=True)
                continue
            
            try:
                if self.pipeline is not None:
                    sequence_data = [self.pipeline(item) for item in sequence_data]
            except (PIL.UnidentifiedImageError, OSError, FileNotFoundError, botocore.exceptions.ClientError):
                print(f"invalid data: {sample['scenario_id']}", flush=True)
                continue

            yield sequence_data
            del sequence_data