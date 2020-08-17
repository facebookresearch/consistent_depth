#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from typing import Tuple


def reproject(pts3d: np.ndarray, extr: np.ndarray) -> np.ndarray:
    assert pts3d.shape[0] == extr.shape[0] and pts3d.shape[0] == 3
    p_dim, _ = pts3d.shape
    R, t = extr[:, :p_dim], extr[:, -1:]
    return R.T.dot(pts3d - t)


def focal_length(intr: np.ndarray):
    return intr[:2]


def principal_point(intrinsics):
    """
    Args:
        intrinsics: (fx, fy, cx, cy)
    """
    return intrinsics[2:]
    # # center version
    # H, W = shape
    # return torch.tensor(((W - 1) / 2.0, (H - 1) / 2.0), device=_device)


def project(pts3d: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """
    Args:
        pts3d   (3, N)
        intr    (4)

    Returns:
        pixels  (2, N)
    """
    rays = pts3d / -pts3d[-1:]
    fxy = focal_length(intr)
    uvs = rays[:2] * fxy.reshape(-1, 1)
    cs = principal_point(intr)
    # to pixels: (i, j) = (u, -v) + (cx, cy)
    uvs[1] = -uvs[1]
    pixels = uvs + cs.reshape(-1, 1)
    return pixels


def sample(depth: np.ndarray, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        depth   (H, W)
        pixels  (2, N)

    Returns:
        depths  (M): depths at corresponding pixels with nearest neighbour sampling,
                    M <= N, because some depth can be invalid
        ix      (N): whether a pixels[:, i] is inside the image
    """
    pixels_nn = (pixels + 0.5).astype(int)
    H, W = depth.shape
    ix = np.all(
        (
            0 <= pixels_nn[0], pixels_nn[0] <= W - 1,
            0 <= pixels_nn[1], pixels_nn[1] <= H - 1,
        ),
        axis=0,
    )
    pixels_valid = pixels_nn[:, ix]
    indices = pixels_valid[1] * W + pixels_valid[0]
    ds = depth.flatten()[indices]
    return ds, ix
