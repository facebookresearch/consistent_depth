#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
from os.path import join as pjoin
from typing import Dict, Tuple
import numpy as np
from . import load_colmap, image_io as tr
from .geometry_np import reproject, project, sample


def store_visible_points_per_image(
    points3D: Dict[int, load_colmap.Point3D]
) -> Dict[int, np.ndarray]:
    """
    returns dictionary where key is the
        image id: int
    and the value is
        3D points (3, N) that are visible in each image
    (Note: currently images do not contain this info, but a list of -1's)
    """
    map_img_to_pt3D = {}
    for _cur_key, cur_point in points3D.items():
        # assert(cur_key == cur_point) # They are the same by design
        for img_id in cur_point.image_ids:
            if img_id in map_img_to_pt3D:
                map_img_to_pt3D[img_id].append(cur_point.xyz)
            else:
                map_img_to_pt3D[img_id] = [cur_point.xyz]

    for img_id, pt_list in map_img_to_pt3D.items():
        map_img_to_pt3D[img_id] = load_colmap.convert_points3D(np.array(pt_list).T)

    return map_img_to_pt3D


def vote_scale(
    scales: np.ndarray, min_percentile_thresh: int = 10, max_percentile_thresh: int = 90
) -> float:
    """
    Note if len(scales) is really small, e.g., len(scales) == 2, it will return nan
    """
    m = np.percentile(scales, min_percentile_thresh)
    M = np.percentile(scales, max_percentile_thresh)
    ix = np.logical_and(m <= scales, scales <= M)
    scale = np.mean(scales[ix])
    return scale


def calibrate_frame_w_sparse_points(
    pts3d: np.ndarray, intr: np.ndarray, extr: np.ndarray, inv_depth: np.ndarray
) -> float:
    """
    Args:
        pts3d   (3, N)
        intr    (4)
        extr    (3, 4)
        depth   (H, W)

    Returns:
        scale: depth * scale = -pts_in_local_camera_coordinate.z
    """
    # points 3d in local camera coordinate
    # FIXME: deal with the case when a point is behind the camera
    pts3d_cam = reproject(pts3d, extr)
    pts2d = project(pts3d_cam, intr)
    inv_depths, ix = sample(inv_depth, pts2d)
    ds = -pts3d[-1, :][ix]  # Note negative sign
    scales = ds * inv_depths
    return vote_scale(scales)


def calibrate_w_sparse_colmap(
    colmap_dir: str, dense_depth_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        colmap_dir: sparse colmap directory containing
            cameras.bin/txt, points3D.bin/txt, images.bin/txt
        dense_depth_dir: folder name for dense depth directory
        scales_fn (optional): dump per frame scale

    Returns:
        Calibrated intrinsics and extrinsics
        intrinsics  (N, 4)
        extrinsics  (N, 3, 4)
        scales      (N)
    """
    cameras, images, points3D = load_colmap.read_model(path=colmap_dir, ext=".bin")
    # compute intrinsics, extrinsics
    depth_names = [
        x for x in os.listdir(dense_depth_dir) if os.path.splitext(x)[-1] == ".raw"
    ]
    size = tr.load_raw_float32_image(pjoin(dense_depth_dir, depth_names[0])).shape[:2][
        ::-1
    ]
    intrinsics, extrinsics = load_colmap.convert_calibration(cameras, images, size)

    # TODO: make the following up to compute the scale a single function
    map_img_to_pt3D = store_visible_points_per_image(points3D)
    ordered_im_ids = load_colmap.ordered_image_ids(images)
    scales = np.empty(intrinsics.shape[0])
    for i, im_id in enumerate(ordered_im_ids):
        if im_id not in map_img_to_pt3D:
            scales[i] = np.nan
            print('[WARNING] %s does not have visible feature point' % images[im_id].name)
        im_name = images[im_id].name
        depth_fn = pjoin(dense_depth_dir, os.path.splitext(im_name)[0] + ".raw")
        inv_depth = tr.load_raw_float32_image(depth_fn)
        pts3D = map_img_to_pt3D[im_id]
        scale = calibrate_frame_w_sparse_points(
            pts3D, intrinsics[i], extrinsics[i], inv_depth
        )
        scales[i] = scale

    mean_scale = scales[~np.isnan(scales)].mean()
    extrinsics[..., -1] /= mean_scale
    return intrinsics, extrinsics, scales
