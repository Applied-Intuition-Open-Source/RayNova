from typing import Tuple, List, Optional
from PIL import Image
import copy

import cv2
import torch
import numpy as np

from nuscenes_tools.nuscenes_utils.box3d_instance import LiDARInstance3DBoxes


# BGR
OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 127, 80),
    "trailer": (255, 140, 0),
    "traffic_barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "vehicle":  (255, 158, 0),
    "cyclist": (220, 20, 60),
    "other": (0, 0, 0),
    "debris": (105, 105, 105),
    "traffic_light": (255, 255, 0),
    "sign": (0, 255, 255),
    "unknown": (0, 0, 0),
}


MAP_PALETTE_VIS = {
    "road_line_solid_single_white": (255, 0, 0), # red
    "road_line_broken_single_white": (255, 165, 0), # orange
    "lane_surface_street": (255, 255, 209), # light yellow
    # "road_edge_sidewalk": (0, 0, 255), # blue
    # "road_edge_boundary": (135, 206, 255), # skyblue
    # "lane_surface_unstructure": (191, 62, 255), # purple
    # "crosswalk": (0, 255, 0), # green
    "road_line_solid_single_yellow": (255, 0, 0), # yellow
}

# MAP_PALETTE = {
#     "road_line_solid_single_white": (255, 0, 0), # red
#     "road_line_broken_single_white": (255, 165, 0), # orange
#     "lane_surface_street": (255, 255, 209), # light yellow
#     "road_edge_sidewalk": (0, 0, 255), # blue
#     "road_edge_boundary": (135, 206, 255), # skyblue
#     "lane_surface_unstructure": (191, 62, 255), # purple
#     "crosswalk": (0, 255, 0), # green
#     "road_line_solid_single_yellow": (255, 255, 0), # yellow
# }

# OBJECT_PALETTE = {
#     "VEHICLE": (255, 158, 0),
#     "CYCLIST": (220, 20, 60),
#     "TRAFFIC_CONE": (47, 79, 79),
#     "PEDESTRIAN": (0, 0, 230),
#     "TRAFFIC_BARRIER": (112, 128, 144),
# }


def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Args:
        points (torch.Tensor): Points of shape (N, M, 3).
        angles (torch.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns:
        torch.Tensor: Rotated points in shape (N, M, 3)
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, -rot_sin, zeros]),
                torch.stack([rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = torch.stack(
            [
                torch.stack([zeros, rot_cos, -rot_sin]),
                torch.stack([zeros, rot_sin, rot_cos]),
                torch.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")

    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


def visualize_camera(
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        
        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int32),
                    coords[index, end].astype(np.int32),
                    color or OBJECT_PALETTE[str(name).lower()],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def show_box_on_views(classes, images: Tuple[Image.Image, ...],
                      boxes: LiDARInstance3DBoxes, labels, transform,
                      aug_matrix=None):
    # in `third_party/bevfusion/mmdet3d/datasets/nuscenes_dataset.py`, they use
    # (0.5, 0.5, 0) as center, however, visualize_camera assumes this center.
    # bboxes_trans = box_center_shift(boxes, (0.5, 0.5, 0.5))
    
    bboxes_trans = boxes

    vis_output = []
    for idx, img in enumerate(images):
        image = np.asarray(img)
        # the color palette for `visualize_camera` is RGB, but they draw on BGR.
        # So we make our image to BGR. This can match their color.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        trans = transform[idx]
        if aug_matrix is not None:
            trans = aug_matrix[idx] @ trans
        # mmdet3d can only save image to file.
        img_out = visualize_camera(
            image=image, bboxes=bboxes_trans, labels=labels,
            transform=trans, classes=classes, thickness=1,
        )
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        vis_output.append(Image.fromarray(img_out))
    return vis_output


def draw_box_on_imgs(data, ori_imgs, classes, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg or ori_imgs is None:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    gt_bboxes_3d = data['gt_bboxes_3d']
    out_imgs = show_box_on_views(
        classes, in_imgs, gt_bboxes_3d, data['gt_labels_3d'].numpy(),
        data['lidar2image'].numpy(), data['img_aug_matrix'].numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs


def visualize_points(image, points, labels, transform, classes):
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    
    coords = np.concatenate(
        [points, np.ones_like(points[..., :1])], axis=-1
    )
    transform = copy.deepcopy(transform).reshape(4, 4)
    coords = coords @ transform.T

    indices = np.any(coords[..., 2] > 0, axis=1)
    coords = coords[indices]
    labels = labels[indices]

    indices = np.argsort(-np.min(coords[..., 2], axis=1))
    coords = coords[indices]
    labels = labels[indices]

    coords[..., 2] = np.clip(coords[..., 2], a_min=1e-5, a_max=1e5)
    coords[..., 0] /= coords[..., 2]
    coords[..., 1] /= coords[..., 2]
    
    for index in range(coords.shape[0]):
        name = classes[labels[index]]
        for pid in range(coords.shape[1]):
            if coords[index, pid, 2] > 0.1:
                if str(name).lower() in MAP_PALETTE_VIS:
                    color = MAP_PALETTE_VIS[str(name).lower()]
                else:
                    continue
                cv2.circle(canvas, coords[index, pid, :2].astype(np.int32), 1, color, -1)
        
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def show_map_points_on_views(classes, images: Tuple[Image.Image, ...],
                      points: torch.Tensor, labels, transform,
                      aug_matrix=None):
    
    vis_output = []
    for idx, img in enumerate(images):
        image = np.asarray(img)
        # the color palette for `visualize_camera` is RGB, but they draw on BGR.
        # So we make our image to BGR. This can match their color.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        trans = transform[idx]
        if aug_matrix is not None:
            trans = aug_matrix[idx] @ trans
            
        img_out = visualize_points(image, points, labels, trans, classes)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        vis_output.append(Image.fromarray(img_out))
    return vis_output



def draw_map_points_on_imgs(data, ori_imgs, classes, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg or ori_imgs is None:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs

    map_points = data['map_sampled_points']
    out_imgs = show_map_points_on_views(
        classes, in_imgs, map_points, data['map_type_labels'].numpy(),
        data['lidar2image'].numpy(), data['img_aug_matrix'].numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs


# def box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
#     raw_data = bboxes.tensor.numpy()
#     new_bboxes = LiDARInstance3DBoxes(
#         raw_data, box_dim=raw_data.shape[-1], origin=new_center)
#     return new_bboxes


# def trans_boxes_to_views(bboxes, transforms, aug_matrixes=None, proj=True):
#     """This is a wrapper to perform projection on different `transforms`.

#     Args:
#         bboxes (LiDARInstance3DBoxes): bboxes
#         transforms (List[np.arrray]): each is 4x4.
#         aug_matrixes (List[np.array], optional): each is 4x4. Defaults to None.

#     Returns:
#         List[np.array]: each is Nx8x3, where z always equals to 1 or -1
#     """
#     if len(bboxes) == 0:
#         return None

#     coords = []
#     for idx in range(len(transforms)):
#         if aug_matrixes is not None:
#             aug_matrix = aug_matrixes[idx]
#         else:
#             aug_matrix = None
#         coords.append(
#             trans_boxes_to_view(bboxes, transforms[idx], aug_matrix, proj))
#     return coords


# def trans_boxes_to_view(bboxes, transform, aug_matrix=None, proj=True):
#     """2d projection with given transformation.

#     Args:
#         bboxes (LiDARInstance3DBoxes): bboxes
#         transform (np.array): 4x4 matrix
#         aug_matrix (np.array, optional): 4x4 matrix. Defaults to None.

#     Returns:
#         np.array: (N, 8, 3) normlized, where z = 1 or -1
#     """
#     if len(bboxes) == 0:
#         return None

#     bboxes_trans = box_center_shift(bboxes, (0.5, 0.5, 0.5))
#     trans = transform
#     if aug_matrix is not None:
#         aug = aug_matrix
#         trans = aug @ trans
#     corners = bboxes_trans.corners
#     num_bboxes = corners.shape[0]

#     coords = np.concatenate(
#         [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
#     )
#     trans = copy.deepcopy(trans).reshape(4, 4)
#     coords = coords @ trans.T

#     coords = coords.reshape(-1, 4)
#     # we do not filter > 0, need to keep sign of z
#     if proj:
#         z = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
#         coords[:, 0] /= z
#         coords[:, 1] /= z
#         coords[:, 2] /= np.abs(coords[:, 2])

#     coords = coords[..., :3].reshape(-1, 8, 3)
#     return coords