# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision
import collections
from PIL import Image
import cv2
import numpy as np
import functools
from pyquaternion import Quaternion
from .box3d_instance import BaseInstance3DBoxes
from .points_instance import BasePoints
from .data_container import DataContainer

# for training on clusters, we need to change the loading pipeline
import sys
sys.path.append("../..")

def to_tensor(data) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper


def read_image(filename, cam_data):    
    img = Image.open(filename, mode='r')
    img_size = img.size

    if 'distortion' in cam_data:
        new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(cam_data['cam_intrinsic'], cam_data['distortion'], img_size, 0, img_size)
        
        img = cv2.undistort(np.array(img), cam_data['cam_intrinsic'], cam_data['distortion'], None, new_intrinsic)
        cam_data['cam_intrinsic'] = new_intrinsic
        img = Image.fromarray(img)

    return img # PIL Image, [H, W, 3]


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DataContainer(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DataContainer(to_tensor(img), stack=True)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths',
                'map_type_labels', 'map_sampled_points',
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DataContainer([to_tensor(res) for res in results[key]])
            else:
                results[key] = DataContainer(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = DataContainer(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DataContainer(
                    to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = DataContainer(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DataContainer(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


class Collect3D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'cam_sweep_ids',
                            'sequence_group_idx', 'curr_to_prev_lidar_rt',
                            'curr_to_first_lidar_rt', 'img_path',
                            'start_of_sequence', 'index', 'global_to_curr_lidar_rt',
                            'prev_lidar_to_global_rt', 'sample_index', 'description',
                            'location', 'timeofday', 'timestep', 'language_description')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DataContainer(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, map_names=None, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.map_names = map_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DataContainer(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DataContainer(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_names' in results:
                assert self.class_names is not None, "There must exist class_names for GT label"
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = np.array([
                        'traffic_barrier' if n.lower() == 'barrier' else n for n in results['gt_names_3d']
                    ])
                    
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n.lower())
                        for n in results['gt_names_3d']
                    ],
                            dtype=np.int64)
    
    
        if 'map_type_names' in results and len(results['map_type_names']) == 0:
            results['map_type_labels'] = np.array([], dtype=np.int64)
        elif 'map_type_names' in results:
            results['map_type_labels'] = np.array([
                self.map_names.index(n.lower())
                for n in results['map_type_names']
            ], dtype=np.int64)
    
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str


class LoadMultiViewImageFromFiles_BEVDet(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False,
                 sequential=False, aligned=False, trans_only=True):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=data_config['mean'],std=data_config['std'])))
        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        if isinstance(resize, float):
            post_rot *= resize
        else:
            post_rot = torch.Tensor([[resize[0], 0], [0, resize[1]]]) @ post_rot
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        keep_ratio = self.data_config.get('keep_ratio', True)
        if self.is_train:
            if keep_ratio:
                resize = float(fW)/float(W)
                resize += np.random.uniform(*self.data_config['resize'])
                resize_dims = (int(W * resize), int(H * resize))
            else:
                resize_w, resize_h = float(fW)/float(W), float(fH)/float(H)
                resize_w += np.random.uniform(*self.data_config['resize'])
                resize_h += np.random.uniform(*self.data_config['resize'])
                resize_dims = (int(W * resize_w), int(H * resize_h))                
                resize = (resize_w, resize_h)


            newW, newH = resize_dims
            if self.data_config['crop_h'][0] < 0:
                crop_h = int(np.random.uniform(0, max(0, newH - fH)))
            else:
                crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            if keep_ratio: 
                resize = float(fW)/float(W)
                resize += self.data_config.get('resize_test', 0.0)
                if scale is not None:
                    resize = scale
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
            else:
                resize_w, resize_h = float(fW)/float(W), float(fH)/float(H)
                resize_w += self.data_config.get('resize_test', 0.0)
                resize_h += self.data_config.get('resize_test', 0.0)
                if scale is not None:
                    if isinstance(scale, float):
                        resize_w = scale
                        resize_h = scale
                    else:
                        resize_w, resize_h = scale
                resize_dims = (int(W * resize_w), int(H * resize_h))
                newW, newH = resize_dims
                resize = (resize_w, resize_h)
            if self.data_config['crop_h'][0] < 0:
                crop_h = newH - fH
            else:
                crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_inputs(self,results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        for cam in cams:
            cam_data = results['img_info'][cam]

            if 'data_path' in cam_data:
                filename = cam_data['data_path']
                if cam_data.get("load_image", True):
                    img = read_image(filename, cam_data)
                else:
                    img = Image.fromarray(np.zeros((self.data_config['src_size'][0], self.data_config['src_size'][1], 3), dtype=np.uint8))
            elif 'data' in cam_data:
                img = cam_data['data']
            else:
                raise ValueError(f"Invalid cam_data: {cam_data}")

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = read_image(filename_adjacent, results['adjacent']['cams'][cam])
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = read_image(filename_adjacent, results['adjacent']['cams'][cam])
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) @ egoadj2global @ lidar2ego
                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))
        return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results