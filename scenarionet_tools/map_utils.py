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

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import triangulate, split

def sample_points_from_polygon(polygon, n):
    if isinstance(polygon, np.ndarray):
        polygon = Polygon(polygon[..., :2])
    tris = triangulate(polygon)

    tris = [t for t in tris if t.area > 1e-12 and polygon.contains(t.representative_point()) or polygon.intersects(t)]
    areas = np.array([t.area for t in tris])
    probs = areas / areas.sum()

    # 采样每个三角形内的随机点（用重心坐标方法）
    chosen_idx = np.random.choice(len(tris), size=n, p=probs)
    pts = []
    for idx in chosen_idx:
        tri = tris[idx]
        xys = np.array(tri.exterior.coords)[:3]  # 三角形的三个顶点 (x,y)
        v0, v1, v2 = xys[0], xys[1], xys[2]
        # 生成两个随机数并转换为均匀三角形内的点
        r1 = np.random.rand()
        r2 = np.random.rand()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
        pts.append(tuple(point))
    return np.array(pts)

def sample_points_from_polyline(points, n_samples=1):
    seg_vecs = points[1:] - points[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum_lens = np.cumsum(seg_lens)
    total_len = cum_lens[-1]

    dists = np.random.rand(n_samples) * total_len

    samples = []
    for d in dists:
        i = np.searchsorted(cum_lens, d)
        prev_len = cum_lens[i-1] if i > 0 else 0.0
        t = (d - prev_len) / seg_lens[i]
        p = points[i] + t * seg_vecs[i]
        samples.append(p)
    
    return np.array(samples)


def resample_polyline(points, step=10.0):
    """
    Resample a (N,3) polyline so that consecutive points are exactly `step` meters apart,
    except possibly the final segment.

    Args:
        points (np.ndarray): (N,3) original points
        step (float): target spacing (10 meters)

    Returns:
        np.ndarray: resampled polyline
    """
    points = np.asarray(points) # (N, 3)
    diffs = np.diff(points, axis=0) # (N-1, 3)
    seg_lengths = np.linalg.norm(diffs, axis=1) # (N-1,)

    # Cumulative length for the polyline
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cumlen[-1]

    # Target sample positions
    sample_positions = np.arange(0, total_length, step) # (M,)
    sample_positions = np.concatenate([sample_positions, np.array([total_length])], axis=0)

    # Output sampled points
    segments = []

    # For each target distance, find which segment it lies in
    prev_idx = 1
    prev_p = points[0]
    for s in sample_positions[1:]:
        # Find segment index
        idx = np.searchsorted(cumlen, s) - 1
        idx = max(0, min(idx, len(points) - 2))

        # Interpolation factor
        seg_start = cumlen[idx]
        seg_end = cumlen[idx + 1]
        t = (s - seg_start) / (seg_end - seg_start)

        # Linear interpolation
        p = points[idx] * (1 - t) + points[idx + 1] * t

        segment = points[prev_idx:idx+1]
        segment = np.concatenate([prev_p[None], segment, p[None]], axis=0)
        segments.append(segment)
        prev_idx = idx + 1
        prev_p = p
    # Add last original point
    return segments


def split_polygon_max_area(poly, max_area = 20):
    """
    Recursively split polygon until all pieces have area <= max_area.
    Returns a list of Polygon pieces.
    """
    if isinstance(poly, np.ndarray):
        poly = Polygon(poly[..., :2])
    pieces = [poly]
    result = []

    while pieces:
        p = pieces.pop()

        if p.area <= max_area:
            result.append(p)
            continue

        # Compute bounding box
        minx, miny, maxx, maxy = p.bounds

        # split direction = longest bbox axis
        if (maxx - minx) >= (maxy - miny):
            split_coord = 0.5 * (minx + maxx)
            cutting_line = LineString([
                (split_coord, miny - 10), 
                (split_coord, maxy + 10)
            ])
        else:
            split_coord = 0.5 * (miny + maxy)
            cutting_line = LineString([
                (minx - 10, split_coord), 
                (maxx + 10, split_coord)
            ])

        # Try splitting
        try:
            splitted = split(p, cutting_line)
        except Exception:
            # Split failed → keep original
            result.append(p)
            continue

        # FIX: GeometryCollection 需要过滤出 polygon
        sub_polygons = []
        for geom in splitted.geoms:
            if geom.is_empty:
                continue
            if isinstance(geom, Polygon):
                sub_polygons.append(geom)
            elif geom.geom_type == "MultiPolygon":
                sub_polygons.extend(list(geom.geoms))  # flatten

        # If no valid polygons, keep original
        if not sub_polygons:
            result.append(p)
            continue

        # Push new pieces to stack
        pieces.extend(sub_polygons)

    return result


def convert_points(points, rotation, translation, ground_z=-0.3186, default_z=0):
    # points: (N, 2) or (N, 3)
    if points.shape[-1] == 2:
        use_default_z = True
        points = np.concatenate([points, np.full((points.shape[0], 1), default_z)], axis=-1)

    else:
        use_default_z = False
    points = points @ rotation.T + translation[None]
    if use_default_z:
        points = np.concatenate([points[..., :2], np.full((points.shape[0], 1), ground_z)], axis=-1)
    return points

def filter_map_elements(points):
    dist = np.linalg.norm(points[..., :2], axis=1)
    dist = dist.min()
    return dist < 50


def fit_plane_from_points(points):
    # points: (N, 3)
    # Fit a plane z = a*x + b*y + c from points
    
    points = np.asarray(points)
    
    X = points[:, :2]  # x, y
    X = np.concatenate([X, np.ones((points.shape[0], 1))], axis=-1)  # add constant term
    z = points[:, 2]
    
    # least squares to solve plane parameters [a, b, c]
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    return coeffs


def compute_z(coeffs, xy_points):
    """
    compute z from plane parameters
    xy_points: shape (N, 2)
    """
    a, b, c = coeffs
    xy_points = np.asarray(xy_points)
    z = a * xy_points[:, 0] + b * xy_points[:, 1] + c
    return z