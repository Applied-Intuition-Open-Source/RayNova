# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from abc import abstractmethod
from torch import Tensor
from typing import Optional, Tuple, Union

from .points_instance import BasePoints
from .box_utils import limit_period, rotation_3d_in_axis, xywhr2xyxyr


def get_box_type(box_type):
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure.
            The valid value are "LiDAR", "Camera", or "Depth".

    Returns:
        tuple: Box type and box mode.
    """
    box_type_lower = box_type.lower()
    if box_type_lower == 'lidar':
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = None
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth"'
                         f' are supported, got {box_type}')

    return box_type_3d, box_mode_3d


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Default to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Default to True.
        origin (tuple[float]): The relative position of origin in the box.
            Default to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(
                dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """torch.Tensor: Corners of each box with size (N, 8, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """torch.Tensor: A vector with height of each box."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """torch.Tensor: A vector with the top height of each box."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """torch.Tensor: A vector with bottom's height of each box."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """Calculate the center of all the boxes.

        Note:
            In the MMDetection3D's convention, the bottom center is
            usually taken as the default center.

            The relative position of the centers in different kinds of
            boxes are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
            It is recommended to use ``bottom_center`` or ``gravity_center``
            for more clear usage.

        Returns:
            torch.Tensor: A tensor with center of each box.
        """
        return self.bottom_center

    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: a tensor with 8 corners of each box."""
        pass

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the boxes in BEV along given BEV direction."""
        pass

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (torch.Tensor): Translation vector of size 1x3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each box is \
                inside the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 1] > box_range[1])
                          & (self.tensor[:, 2] > box_range[2])
                          & (self.tensor[:, 0] < box_range[3])
                          & (self.tensor[:, 1] < box_range[4])
                          & (self.tensor[:, 2] < box_range[5]))
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each box is inside \
                the reference range.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type \
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw.
            period (float): The expected period.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: float = 0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Args:
            threshold (float): The threshold of minimal sizes.

        Returns:
            torch.Tensor: A binary vector which represents whether each \
                box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = ((size_x > threshold)
                & (size_y > threshold) & (size_z > threshold))
        return keep

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of  \
                :class:`BaseInstances3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim,
                with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list):
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (list[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated Boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(
            torch.cat([b.tensor for b in boxes_list], dim=0),
            box_dim=boxes_list[0].tensor.shape[1],
            with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the \
                specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw)

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties \
                as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        heighest_of_bottom = torch.max(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    # @classmethod
    # def overlaps(cls, boxes1, boxes2, mode='iou'):
    #     """Calculate 3D overlaps of two boxes.

    #     Note:
    #         This function calculates the overlaps between ``boxes1`` and
    #         ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

    #     Args:
    #         boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
    #         boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
    #         mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

    #     Returns:
    #         torch.Tensor: Calculated iou of boxes' heights.
    #     """
    #     assert isinstance(boxes1, BaseInstance3DBoxes)
    #     assert isinstance(boxes2, BaseInstance3DBoxes)
    #     assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
    #         f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

    #     assert mode in ['iou', 'iof']

    #     rows = len(boxes1)
    #     cols = len(boxes2)
    #     if rows * cols == 0:
    #         return boxes1.tensor.new(rows, cols)

    #     # height overlap
    #     overlaps_h = cls.height_overlaps(boxes1, boxes2)

    #     # obtain BEV boxes in XYXYR format
    #     boxes1_bev = xywhr2xyxyr(boxes1.bev)
    #     boxes2_bev = xywhr2xyxyr(boxes2.bev)

    #     # bev overlap
    #     overlaps_bev = boxes1_bev.new_zeros(
    #         (boxes1_bev.shape[0], boxes2_bev.shape[0])).cuda()  # (N, M)
    #     iou3d_cuda.boxes_overlap_bev_gpu(boxes1_bev.contiguous().cuda(),
    #                                      boxes2_bev.contiguous().cuda(),
    #                                      overlaps_bev)

    #     # 3d overlaps
    #     overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

    #     volume1 = boxes1.volume.view(-1, 1)
    #     volume2 = boxes2.volume.view(1, -1)

    #     if mode == 'iou':
    #         # the clamp func is used to avoid division of 0
    #         iou3d = overlaps_3d / torch.clamp(
    #             volume1 + volume2 - overlaps_3d, min=1e-8)
    #     else:
    #         iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

    #     return iou3d

    def new_box(self, data):
        """Create a new box object with data.

        The new box and its tensor has the similar properties \
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, \
                the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                 up z    x front (yaw=0)
                                    ^   ^
                                    |  /
                                    | /
        (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2. The yaw is 0 at
    the positive direction of x axis, and increases from the positive direction
    of x to the positive direction of y.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 2

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y <------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(
            corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        elif bev_direction == 'vertical':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 1] = -points[:, 1]
                elif bev_direction == 'vertical':
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self,
            src=Box3DMode.LIDAR,
            dst=dst,
            rt_mat=rt_mat,
            correct_yaw=correct_yaw)

    def enlarged_box(
            self, extra_width: Union[float, Tensor]) -> 'LiDARInstance3DBoxes':
        """Enlarge the length, width and height of boxes.

        Args:
            extra_width (float or Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)