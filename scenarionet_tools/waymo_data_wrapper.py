import torch
import os
import io
import sys
import random
import pickle
import copy
import PIL
import time
import copy
import types
import importlib
import lance
import lance.torch.data
import pyarrow.compute as pc
from typing import TypedDict, Dict, List, Optional, Tuple, Union
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from lance.sampler import ShardedFragmentSampler, ShardedBatchSampler
from functools import partial
import json
from metadrive.scenario import ScenarioDescription
from PIL import Image
from collections import OrderedDict

from nuscenes_tools.nuscenes_utils.pipelines import Compose
import nuscenes_tools.nuscenes_utils.pipelines as pipelines
from nuscenes_tools.nuscenes_utils.box3d_instance import LiDARInstance3DBoxes
import scenarionet_tools.map_utils as map_utils

from shapely.geometry import Polygon, Point
from shapely.ops import triangulate

import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2


# os.environ['AWS_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com'
# os.environ['AWS_ACCESS_KEY_ID'] = 'fd082c21e10475d60adc85b01be246745860a21a'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'rEW3zxhO6sV802Rg7EqgfA+CLTGh0eqHu0cIcgujauw='
# os.environ['URSA_SDK_GRPC_HOSTNAME'] = 'grpc.neuron.oci.applied.dev'
# os.environ['AWS_DEFAULT_REGION'] = 'us-phoenix-1'


# os.environ['DATASET_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com'
# os.environ['DATASET_ACCESS_KEY_ID'] = 'fd082c21e10475d60adc85b01be246745860a21a'
# os.environ['DATASET_SECRET_ACCESS_KEY'] = 'rEW3zxhO6sV802Rg7EqgfA+CLTGh0eqHu0cIcgujauw='
# os.environ['DATASET_REGION'] = 'us-phoenix-1'

class_names = [
    'unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist',
]

ego_offset = np.array([
    [1, 0, 0, -1.5],
    [0, 1, 0, 0],
    [0, 0, 1, -1.5],
    [0, 0, 0, 1],
])


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


class WaymoDataWrapper(torch.utils.data.Dataset):
    def __init__(self, data_config, pipeline=None):
        # load image:  0 (w/o image), 1 (first frame only), 2 (all images)
        self.data_config = data_config
        self.data_path = data_config['dataset_path']
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
            
        self.all_contexts = self._load_contexts()
    
    def _read_context(self, tag, context_name):
        paths = tf.io.gfile.glob(f'{self.data_path}/{tag}/{context_name}.parquet')
        return dd.read_parquet(paths)

    def _load_contexts(self):
        stats_dir = os.path.join(self.data_path, 'stats')
        stats_files = [f for f in os.listdir(stats_dir) if f.endswith('.parquet')]
        contexts = [f[:-8] for f in stats_files]
        return contexts
    
    def __len__(self):
        return len(self.all_contexts)

    def _get_scenario_info(self):
        scenario_dict = {}
        scenario_dict['location'] = None
        scenario_dict['timeofday'] = None
        scenario_dict['language_description'] = 'This is a driving scene at California Bay Area or Phoenix.'

        return scenario_dict
    
    def _get_object_info(self, frames_object, timestamp):
        gt_bboxes_3d = []
        gt_names_3d = []
        
        if timestamp in frames_object:
            positions = []
            sizes = []
            headings = []
            labels = []
            frame_object = frames_object[timestamp]
            for obj_info in frame_object:
                cx = obj_info['[LiDARBoxComponent].box.center.x']
                cy = obj_info['[LiDARBoxComponent].box.center.y']
                cz = obj_info['[LiDARBoxComponent].box.center.z']
                length = obj_info['[LiDARBoxComponent].box.size.x']
                width = obj_info['[LiDARBoxComponent].box.size.y']
                height = obj_info['[LiDARBoxComponent].box.size.z']
                heading = obj_info['[LiDARBoxComponent].box.heading']
                label = obj_info['[LiDARBoxComponent].type']
                
                positions.append([cx, cy, cz])
                sizes.append([length, width, height])
                headings.append([heading])
                labels.append(label)
            positions = np.array(positions)
            sizes = np.array(sizes)
            headings = np.array(headings)
            
            positions = positions + ego_offset[:3, 3][None]
            
            
            positions = np.array(positions).astype(np.float32)
            sizes = np.array(sizes).astype(np.float32)
            headings = np.array(headings).astype(np.float32)

            for obj_id in range(positions.shape[0]):
                bbox = np.concatenate([positions[obj_id], sizes[obj_id], headings[obj_id]], axis=0)
                

                label = class_names[labels[obj_id]]
                if label == 'sign':
                    continue
                
                gt_bboxes_3d.append(bbox)
                gt_names_3d.append(label)
        
        if len(gt_bboxes_3d) > 0:
            gt_bboxes_3d = np.stack(gt_bboxes_3d, axis=0)
        else:
            gt_bboxes_3d = np.zeros((0, 7))
        gt_names_3d = np.array(gt_names_3d)

        return {'gt_bboxes_3d': gt_bboxes_3d, 'gt_names_3d': gt_names_3d}
        
    
    def _get_camera_info(self, frame_info, id, camera_names):
        all_camera_info = {}
        for cam_id, frame_cam_info in enumerate(frame_info):
            cam_info = {}
            img_bytes = frame_cam_info['[CameraImageComponent].image']
            raw_image = Image.open(io.BytesIO(img_bytes))
            cam_info['data'] = raw_image
            
            cam_intrinsic = np.eye(3)
            cam_intrinsic[0, 0] = frame_cam_info['[CameraCalibrationComponent].intrinsic.f_u']
            cam_intrinsic[1, 1] = frame_cam_info['[CameraCalibrationComponent].intrinsic.f_v']
            cam_intrinsic[0, 2] = frame_cam_info['[CameraCalibrationComponent].intrinsic.c_u']
            cam_intrinsic[1, 2] = frame_cam_info['[CameraCalibrationComponent].intrinsic.c_v']
                        
            
            cam_extrinsic = frame_cam_info['[CameraCalibrationComponent].extrinsic.transform']
            cam_extrinsic = cam_extrinsic.reshape(4, 4).astype(np.float32)
            cam_transpose = np.array([
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ])
            cam_extrinsic = ego_offset @cam_extrinsic @ cam_transpose
            
            cam_translation = cam_extrinsic[:3, 3]
            cam_rotation = cam_extrinsic[:3, :3]
            
            ego2global = frame_cam_info['[CameraImageComponent].pose.transform']
            ego2global = ego2global.reshape(4, 4).astype(np.float32)
            ego_vehicle_rotation = ego2global[:3, :3]
            ego_vehicle_translation = ego2global[:3, 3]

            cam_info['cam_intrinsic'] = np.array(cam_intrinsic, dtype=np.float32)
            cam_info['sensor2lidar_translation'] = np.array(cam_translation, dtype=np.float32)
            cam_info['sensor2lidar_rotation'] = np.array(cam_rotation, dtype=np.float32)

            cam_info['ego2global_rotation'] = np.array(ego_vehicle_rotation, dtype=np.float32)
            cam_info['ego2global_translation'] = np.array(ego_vehicle_translation, dtype=np.float32)
            
            global2lidar = np.linalg.inv(ego2global)
            cam_info['global2lidar_rotation'] = global2lidar[:3, :3]
            cam_info['global2lidar_translation'] = global2lidar[:3, 3]
            
            cam_name = camera_names[cam_id]
            all_camera_info[cam_name] = cam_info

        return all_camera_info

    def _get_sensor_sequence_info(self, context_name):
        cam_calib_df = self._read_context('camera_calibration', context_name)
        cam_img_df = self._read_context('camera_image', context_name)
        
        cam_img_df = v2.merge(cam_calib_df, cam_img_df)

        # Join all DataFrames using matching columns
        cam_img_df.head()
        cam_img_df = cam_img_df.compute()
        # frames = cam_image_w_box_df.groupby("key.frame_timestamp_micros").apply(lambda g: g.to_dict("records")).reset_index(name="items")
        # data_info.update(scenario)
        # data_info['sample_idx'] = item['scenario_id']
        frames = [
            rows.to_dict("records") for group_key, rows in cam_img_df.groupby("key.frame_timestamp_micros")
        ]

        association_df = self._read_context('camera_to_lidar_box_association', context_name)
        cam_object_df = self._read_context('camera_box', context_name)
        cam_object_df = v2.merge(association_df, cam_object_df)
        lidar_box_df = self._read_context('lidar_box', context_name)
        obj_df = v2.merge(cam_object_df, lidar_box_df, left_nullable=True)
        obj_df.head()
        obj_df = obj_df.compute()
        frames_object = {
            group_key: rows.to_dict("records") for group_key, rows in obj_df.groupby("key.frame_timestamp_micros")
        }

        horizon = self.data_config['horizon']
        if horizon == 'all':
            assert not self.data_config['random_interval']
            horizon = len(frames) // self.data_config['interval']
        intervals = []
        if self.data_config['random_interval']:
            intervals = [random.randint(0, self.data_config['interval']) for _ in range(horizon-1)]
        else:
            intervals = [self.data_config['interval']] * (horizon - 1)
        
        intervals = [0] + intervals
        
        total_len = sum(intervals)
        sids = list(range(len(frames)-total_len))
        sid = random.choice(sids)
        if 'first_frame' in self.data_config:
            sid = self.data_config['first_frame']

        timesteps = np.cumsum(intervals)
        seq_ids = timesteps + sid
        assert len(seq_ids) == horizon and seq_ids[-1] < len(frames)

        image_dict_sequence = []
        bbox_sequence = []
        action_sequence = []
        for id_in_seq, id in enumerate(seq_ids):
            frame_info = frames[id]
            timestamp = frame_info[0]['key.frame_timestamp_micros']
            all_camera_info = self._get_camera_info(frame_info, id_in_seq, self.data_config['cams'])
            image_dict_sequence.append(all_camera_info)

            if 'load_bbox' in self.data_config and self.data_config['load_bbox']:
                all_object_info = self._get_object_info(frames_object, timestamp)
                bbox_sequence.append(all_object_info)
            
        if horizon > 1:
            timesteps = timesteps / ((horizon - 1) * 10)
        else:
            timesteps = np.zeros_like(timesteps)
        return image_dict_sequence, bbox_sequence, action_sequence, timesteps, seq_ids

    def _filter_bbox(self, xyz):
        obj_range = self.data_config['object_range']
        flag = (xyz[:, 0] > obj_range[0]) & (xyz[:, 1] > obj_range[1]) & (xyz[:, 2] > obj_range[2]) \
            & (xyz[:, 0] < obj_range[3]) & (xyz[:, 1] < obj_range[4]) & (xyz[:, 2] < obj_range[5])
        return flag        

    def _convert_and_filter_bbox_coordinate(self, bbox_sequence, image_dict_sequence):
        cam_name = 'CAM_FRONT'
        first_frame_info = image_dict_sequence[0]
        first_frame_global2lidar_rotation = first_frame_info[cam_name]['global2lidar_rotation']
        first_frame_global2lidar_translation = first_frame_info[cam_name]['global2lidar_translation']
        first_frame_global2lidar = np.eye(4)
        first_frame_global2lidar[:3, :3] = first_frame_global2lidar_rotation
        first_frame_global2lidar[:3, 3] = first_frame_global2lidar_translation

        for frame_box, frame_info in zip(bbox_sequence, image_dict_sequence):
            if len(frame_box['gt_bboxes_3d']) == 0:
                continue
            curr_frame_global2lidar_rotation = frame_info[cam_name]['global2lidar_rotation']
            curr_frame_global2lidar_translation = frame_info[cam_name]['global2lidar_translation']
            
            curr_frame_global2lidar = np.eye(4)
            curr_frame_global2lidar[:3, :3] = curr_frame_global2lidar_rotation
            curr_frame_global2lidar[:3, 3] = curr_frame_global2lidar_translation
            
            curr_to_first = first_frame_global2lidar @ np.linalg.inv(curr_frame_global2lidar)
            curr_to_first_rotation = curr_to_first[:3, :3]
            curr_to_first_translation = curr_to_first[:3, 3]
            
            xyz = frame_box['gt_bboxes_3d'][:, :3]

            flag = self._filter_bbox(xyz)
            frame_box['gt_bboxes_3d'] = frame_box['gt_bboxes_3d'][flag]
            frame_box['gt_names_3d'] = frame_box['gt_names_3d'][flag]

            if len(frame_box['gt_bboxes_3d']) == 0:
                continue
            
            new_xyz, new_heading = convert_bbox(frame_box['gt_bboxes_3d'][:, :3], frame_box['gt_bboxes_3d'][:, 6], curr_to_first_rotation, curr_to_first_translation)
            frame_box['gt_bboxes_3d'][:, :3] = new_xyz
            frame_box['gt_bboxes_3d'][:, 6] = new_heading
            
            frame_box['gt_bboxes_3d'] = LiDARInstance3DBoxes(frame_box['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5))

        return bbox_sequence

    def _get_map_info_placeholder(self, scenario, image_dict_sequence):
        map_sequence = []
        for frame_id, frame_info in enumerate(image_dict_sequence):
            frame_map_type = []
            frame_map_points = []
          
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
                
    def _preprocess_sample(self, context_name):
        data_info = {}

        scenario_dict = self._get_scenario_info()
        data_info.update(scenario_dict)
        
        image_dict_sequence, bbox_sequence, action_sequence, timesteps, seq_ids = self._get_sensor_sequence_info(context_name)
        for camera_infos in image_dict_sequence:
            if len(camera_infos) < self.data_config['Ncams']:
                return None
        
        if 'load_bbox' in self.data_config and self.data_config['load_bbox']:
            bbox_sequence = self._convert_and_filter_bbox_coordinate(bbox_sequence, image_dict_sequence)

        if 'load_map' in self.data_config and self.data_config['load_map']:
            map_sequence = self._get_map_info_placeholder(scenario_dict, image_dict_sequence)
        
        frame_dict_sequence = []
        for fid, camera_infos in enumerate(image_dict_sequence):
            frame_dict = {}
            frame_data_info = data_info.copy()
            description = frame_data_info['language_description']
            sentences = description.split('.')
            if sentences[-1] == '':
                sentences = sentences[:-1]
            description = '.'.join(sentences) + '.'
            if description == '.':
                description = ''
            frame_data_info['description'] = description
                
            frame_dict.update(frame_data_info)
            frame_dict['img_info'] = camera_infos
            frame_dict['timestep'] = timesteps[fid]
            
            if 'load_bbox' in self.data_config and self.data_config['load_bbox']:
                frame_dict.update(bbox_sequence[fid])

            if 'load_action' in self.data_config and self.data_config['load_action']:
                frame_dict.update(action_sequence[fid])
            
            if 'load_map' in self.data_config and self.data_config['load_map']:
                # raise NotImplementedError('Map is not supported for Inhouse dataset.')
                frame_dict.update(map_sequence[fid])
            
            frame_dict_sequence.append(frame_dict)
        
        frame_dict_sequence = self._preprocess_motion(frame_dict_sequence)

        return frame_dict_sequence

    
    def _preprocess_motion(self, frame_dict_sequence):
        cam_name = 'CAM_FRONT'
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
            frame_dict['curr_to_first_lidar_rt' ]= torch.from_numpy(curr_to_first_lidar_rt).float()
        return frame_dict_sequence

    
    def __getitem__(self, index):
        context_name = self.all_contexts[index]

        sequence_data = self._preprocess_sample(context_name)
        if self.pipeline is not None:
            sequence_data = [self.pipeline(item) for item in sequence_data]

        return sequence_data
    