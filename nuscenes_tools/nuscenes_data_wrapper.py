"""IterableDataset wrapper that streams the legacy `.pkl`-packaged nuscenes
annotations and emits per-frame dict sequences in the **same shape** as
`scenarionet_tools.nuplan_data_wrapper.NuPlanDataWrapper`.

The two loaders read different on-disk formats but produce identical
pipeline-ready outputs (same keys, same coordinate conventions), so consumers
that already work with `NuPlanDataWrapper` can swap the loader without
changes.

Output of `__iter__`: `List[frame_dict_after_pipeline]` of length
`data_config['horizon']`. Each frame dict contains (post-pipeline):
`img_inputs`, optional `gt_bboxes_3d` / `gt_labels_3d`, and `img_metas`
(DataContainer) carrying `description`, `location`, `timeofday`,
`sample_idx`, `timestep`, `img_path`, `curr_to_prev_lidar_rt`,
`curr_to_first_lidar_rt`.

Note: legacy nuscenes pkl annotations do not include map features, so this
loader does not support `args.map_condition`.
"""

import os
import os.path as osp
import pickle
import random

import numpy as np
import PIL
import torch
from torch.utils.data import IterableDataset

from infinity.utils.s3_file_utils import load_bytes_file

from .nuscenes_utils.box3d_instance import LiDARInstance3DBoxes
from .nuscenes_utils.pipelines import Compose
from .nuscenes_utils import pipelines
from .nuscenes_utils.utils import nuscenes_get_rt_matrix


_DEFAULT_NUSCENES_CAMS = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT',
)


def build_nuscenes_dataset(args, rank, world_size, source_id=None):
    """Mirrors `scenarionet_tools.nuplan_data_wrapper.build_nuscenes_dataset`
    but produces a `NuScenesDataWrapper` that reads from the legacy pkl
    annotation files."""
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

    data_config = {
        'dataset_name': 'nuscenes',
        'dataset_path': dataset_path,
        'horizon': args.max_horizon,
        # nuscenes keyframes are at 2 Hz; nuplan is at 10 Hz.
        # `args.sample_interval` is in nuplan ticks (10 Hz), so divide by 5.
        'interval': args.sample_interval // 5,
        'random_interval': args.random_sample_interval,

        'load_bbox': args.object_condition,
        'object_range': (-50, -50, -10, 50, 50, 10),

        'cams': list(_DEFAULT_NUSCENES_CAMS),
        'Ncams': 6,
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
        'resize_test': 0.0,

        'load_map': False,  # legacy pkl has no map features
    }

    if args.map_condition:
        raise NotImplementedError(
            'NuScenesDataWrapper (legacy pkl loader) does not provide map '
            'features. Use scenarionet_tools.nuplan_data_wrapper for '
            'map-conditioned nuscenes training.')

    load_keys = ['img_inputs']
    if args.object_condition:
        load_keys.extend(['gt_bboxes_3d', 'gt_labels_3d'])

    if args.object_condition:
        train_pipeline = [
            pipelines.LoadMultiViewImageFromFiles_BEVDet(data_config, is_train=True),
            pipelines.DefaultFormatBundle3D(
                class_names=args.class_names, map_names=None,
                with_gt=True, with_label=True),
            pipelines.Collect3D(keys=load_keys),
        ]
    else:
        train_pipeline = [
            pipelines.LoadMultiViewImageFromFiles_BEVDet(data_config, is_train=True),
            pipelines.DefaultFormatBundle3D(
                class_names=None, map_names=None,
                with_gt=False, with_label=False),
            pipelines.Collect3D(keys=load_keys),
        ]

    return NuScenesDataWrapper(data_config, rank, world_size, train_pipeline)


class NuScenesDataWrapper(IterableDataset):
    """Stream per-frame dict sequences from a legacy nuscenes `.pkl` file.

    Output layout matches `NuPlanDataWrapper.__iter__` exactly so consumers
    don't need to special-case nuscenes vs nuplan."""

    def __init__(self, data_config, rank=0, world_size=1, pipeline=None,
                 ann_file=None, randomize=True, seed=None,
                 random_drop_cond=True):
        super().__init__()
        self.data_config = data_config
        self.data_path = data_config['dataset_path']
        self.dataset_name = 'nuscenes'
        self.rank = rank
        self.world_size = world_size
        self.randomize = randomize
        self.seed = seed
        self.random_drop_cond = random_drop_cond

        self.pipeline = Compose(pipeline) if pipeline is not None else None

        if ann_file is None:
            ann_file = osp.join(self.data_path, 'nuscenes_desc_infos_omnidrive_train.pkl')
        self.ann_file = ann_file
        self.data_infos = self._load_annotations(self.ann_file)
        self.sequence_groups = self._build_sequence_groups(self.data_infos)

    # ---------------------------------------------------------------- IO
    def _load_annotations(self, ann_file):
        data = pickle.load(load_bytes_file(ann_file))
        infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        return infos

    def _build_sequence_groups(self, infos):
        """Bucket frames into scenes. A scene starts at idx 0 or whenever the
        previous frame had no sweeps (legacy heuristic)."""
        groups = []
        curr = []
        for idx, info in enumerate(infos):
            if idx != 0 and len(info.get('sweeps', [])) == 0 and curr:
                groups.append(curr)
                curr = []
            curr.append(idx)
        if curr:
            groups.append(curr)
        return groups

    # ----------------------------------------------------- per-frame dict
    def _get_camera_info(self, info_cams):
        """Project pkl camera entries into the dict shape the BEVDet pipeline
        expects (data_path, cam_intrinsic, sensor2lidar_rotation,
        sensor2lidar_translation, load_image)."""
        all_cam = {}
        for cam_name in self.data_config['cams']:
            if cam_name not in info_cams:
                continue
            cam = info_cams[cam_name]
            all_cam[cam_name] = {
                'data_path': cam['data_path'],
                'cam_intrinsic': np.array(cam['cam_intrinsic'], dtype=np.float32),
                'sensor2lidar_rotation': np.array(cam['sensor2lidar_rotation'], dtype=np.float32),
                'sensor2lidar_translation': np.array(cam['sensor2lidar_translation'], dtype=np.float32),
                'load_image': True,
            }
        return all_cam

    def _get_object_info(self, info):
        """Return per-frame `gt_bboxes_3d` (Nx7, in current-lidar coords) and
        `gt_names_3d` (N,) — matching NuPlanDataWrapper._get_object_info."""
        if 'gt_boxes' not in info or len(info['gt_boxes']) == 0:
            return {
                'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                'gt_names_3d': np.array([], dtype=object),
            }
        if 'valid_flag' in info and info['valid_flag'] is not None:
            mask = info['valid_flag']
        else:
            mask = info.get('num_lidar_pts', np.ones(len(info['gt_boxes']))) > 0

        gt_boxes = np.asarray(info['gt_boxes'])[mask][:, :7].astype(np.float32)
        gt_names = np.asarray(info['gt_names'])[mask]
        return {'gt_bboxes_3d': gt_boxes, 'gt_names_3d': gt_names}

    # --------------------------------------------------- coord transforms
    def _filter_bbox(self, xyz):
        r = self.data_config['object_range']
        return ((xyz[:, 0] > r[0]) & (xyz[:, 1] > r[1]) & (xyz[:, 2] > r[2])
                & (xyz[:, 0] < r[3]) & (xyz[:, 1] < r[4]) & (xyz[:, 2] < r[5]))

    @staticmethod
    def _apply_se3(xyz, heading, T):
        """xyz: (N,3), heading: (N,), T: 4x4. Returns transformed xyz, heading."""
        if len(xyz) == 0:
            return xyz, heading
        Rm = T[:3, :3]
        t = T[:3, 3]
        new_xyz = xyz @ Rm.T + t[None]
        # yaw is rotation around +z; the change is the +z Euler angle of Rm.
        yaw_offset = np.arctan2(Rm[1, 0], Rm[0, 0])
        new_heading = heading + yaw_offset
        return new_xyz, new_heading

    def _convert_and_filter_bbox_coordinate(self, bbox_sequence, info_sequence):
        """Filter each frame's bboxes by the current-lidar object_range, then
        re-express survivors in the first frame's lidar coordinates."""
        first_info = info_sequence[0]
        for frame_box, info in zip(bbox_sequence, info_sequence):
            if len(frame_box['gt_bboxes_3d']) == 0:
                continue

            xyz = frame_box['gt_bboxes_3d'][:, :3]
            flag = self._filter_bbox(xyz)
            frame_box['gt_bboxes_3d'] = frame_box['gt_bboxes_3d'][flag]
            frame_box['gt_names_3d'] = frame_box['gt_names_3d'][flag]
            if len(frame_box['gt_bboxes_3d']) == 0:
                continue

            # current-lidar -> first-lidar
            T = nuscenes_get_rt_matrix(info, first_info, 'lidar', 'lidar')
            new_xyz, new_heading = self._apply_se3(
                frame_box['gt_bboxes_3d'][:, :3],
                frame_box['gt_bboxes_3d'][:, 6],
                T)
            frame_box['gt_bboxes_3d'][:, :3] = new_xyz.astype(np.float32)
            frame_box['gt_bboxes_3d'][:, 6] = new_heading.astype(np.float32)
            frame_box['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                frame_box['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5))
        return bbox_sequence

    # --------------------------------------------------- sequence sampling
    def _sample_sequence(self, rng):
        horizon = self.data_config['horizon']
        interval = self.data_config['interval']
        if self.data_config['random_interval']:
            intervals = [rng.randint(0, max(1, interval)) for _ in range(horizon - 1)]
        else:
            intervals = [interval] * (horizon - 1)
        intervals = [0] + intervals
        total_len = sum(intervals)

        candidate_groups = [g for g in self.sequence_groups if len(g) > total_len]
        if not candidate_groups:
            return None
        group = rng.choice(candidate_groups)
        sids = list(range(len(group) - total_len))
        sid = rng.choice(sids) if sids else 0
        if 'first_frame' in self.data_config:
            sid = self.data_config['first_frame']

        timesteps = np.cumsum(intervals)
        seq_idx = [group[sid + int(t)] for t in timesteps]
        return seq_idx, timesteps

    def _build_sequence_data(self, seq_idx, timesteps):
        info_sequence = [self.data_infos[idx] for idx in seq_idx]
        image_dict_sequence = []
        for info in info_sequence:
            cams = self._get_camera_info(info['cams'])
            if len(cams) < self.data_config['Ncams']:
                return None
            image_dict_sequence.append(cams)

        bbox_sequence = []
        if self.data_config.get('load_bbox'):
            bbox_sequence = [self._get_object_info(info) for info in info_sequence]
            bbox_sequence = self._convert_and_filter_bbox_coordinate(
                bbox_sequence, info_sequence)

        horizon = self.data_config['horizon']
        if horizon > 1:
            ts = timesteps / ((horizon - 1) * 2)  # nuscenes is 2 Hz
        else:
            ts = np.zeros_like(timesteps, dtype=np.float32)

        first_info = info_sequence[0]
        scenario_dict = {
            'sample_idx': first_info.get('token'),
            'location': first_info.get('location'),
            'timeofday': first_info.get('timeofday'),
        }

        frame_dict_sequence = []
        for fid, info in enumerate(info_sequence):
            frame = dict(scenario_dict)
            description = info.get('description') or ''
            if isinstance(description, str) and description:
                sentences = [s for s in description.split('.') if s != '']
                if self.random_drop_cond:
                    sentences = [s for s in sentences if random.random() > 0.25]
                description = ('.'.join(sentences) + '.') if sentences else ''
            frame['description'] = description
            frame['img_info'] = image_dict_sequence[fid]
            frame['img_path'] = {k: v['data_path'] for k, v in image_dict_sequence[fid].items()}
            frame['timestep'] = float(ts[fid])
            if self.data_config.get('load_bbox'):
                frame.update(bbox_sequence[fid])
            frame_dict_sequence.append(frame)

        frame_dict_sequence = self._preprocess_motion(frame_dict_sequence, info_sequence)
        return frame_dict_sequence

    def _preprocess_motion(self, frame_dict_sequence, info_sequence):
        """Attach `curr_to_prev_lidar_rt` and `curr_to_first_lidar_rt` to each
        frame, computed via `nuscenes_get_rt_matrix` (the legacy utility that
        already lives in nuscenes_utils)."""
        first_info = info_sequence[0]
        for fid, frame in enumerate(frame_dict_sequence):
            if fid == 0:
                curr_to_prev = np.eye(4, dtype=np.float32)
                curr_to_first = np.eye(4, dtype=np.float32)
            else:
                curr_to_prev = nuscenes_get_rt_matrix(
                    info_sequence[fid], info_sequence[fid - 1], 'lidar', 'lidar').astype(np.float32)
                curr_to_first = nuscenes_get_rt_matrix(
                    info_sequence[fid], first_info, 'lidar', 'lidar').astype(np.float32)
            frame['curr_to_prev_lidar_rt'] = torch.from_numpy(curr_to_prev).float()
            frame['curr_to_first_lidar_rt'] = torch.from_numpy(curr_to_first).float()
        return frame_dict_sequence

    # ------------------------------------------------ IterableDataset API
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        num_workers = max(num_workers, 1)
        shard_rank = self.rank * num_workers + worker_id

        if self.randomize:
            base_seed = self.seed if self.seed is not None else random.randint(0, 1 << 30)
            rng = random.Random(base_seed + shard_rank)
        else:
            rng = random.Random(shard_rank)

        while True:
            sample = self._sample_sequence(rng)
            if sample is None:
                continue
            seq_idx, timesteps = sample
            sequence_data = self._build_sequence_data(seq_idx, timesteps)
            if sequence_data is None:
                continue

            try:
                if self.pipeline is not None:
                    sequence_data = [self.pipeline(item) for item in sequence_data]
            except (PIL.UnidentifiedImageError, OSError, FileNotFoundError):
                # Skip frames whose image files can't be loaded — same policy
                # as NuPlanDataWrapper.
                continue

            yield sequence_data
            del sequence_data
