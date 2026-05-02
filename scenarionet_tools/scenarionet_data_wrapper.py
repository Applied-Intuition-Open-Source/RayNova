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

import PIL
import torch

from nuscenes_tools.nuscenes_utils.pipelines import Compose  # noqa: F401
import nuscenes_tools.nuscenes_utils.pipelines as pipelines

from scenarionet_tools.nuplan_data_wrapper import NuPlanDataWrapper


_SUMMARY_FILES = {'dataset_summary.pkl', 'dataset_mapping.pkl'}


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
