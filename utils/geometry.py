#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from .torch_helpers import _device
from typing import List


def pixel_grid(batch_size, shape):
    """Returns pixel grid of size (batch_size, 2, H, W).
    pixel positions (x, y) are in range [0, W-1] x [0, H-1]
    top left is (0, 0).
    """
    H, W = shape
    x = torch.linspace(0, W - 1, W, device=_device)
    y = torch.linspace(0, H - 1, H, device=_device)
    Y, X = torch.meshgrid(y, x)
    pixels = torch.stack((X, Y), dim=0)[None, ...]
    return pixels.expand(batch_size, -1, -1, -1)


def principal_point(intrinsics, shape):
    """
    Args:
        intrinsics: (fx, fy, cx, cy)
        shape: (H, W)
    """
    return intrinsics[:, 2:]
    # # center version
    # H, W = shape
    # return torch.tensor(((W - 1) / 2.0, (H - 1) / 2.0), device=_device)


def focal_length(intrinsics):
    return intrinsics[:, :2]


def pixels_to_rays(pixels, intrinsics):
    """Convert pixels to rays in camera space using intrinsics.

    Args:
        pixels (B, 2, H, W)
        intrinsics (B, 4): (fx, fy, cx, cy)

    Returns:
        rays: (B, 3, H, W), where z component is -1, i.e., rays[:, -1] = -1

    """
    # Assume principal point is ((W-1)/2, (H-1)/2).
    B, _, H, W = pixels.shape
    cs = principal_point(intrinsics, (H, W))
    # Convert to [-(W-1)/2, (W-1)/2] x [-(H-1)/2, (H-1)/2)] and bottom left is (0, 0)
    uvs = pixels - cs.view(-1, 2, 1, 1)
    uvs[:, 1] = -uvs[:, 1]  # flip v

    # compute rays (u/fx, v/fy, -1)
    fxys = focal_length(intrinsics).view(-1, 2, 1, 1)
    rays = torch.cat(
        (uvs / fxys, -torch.ones((B, 1, H, W), dtype=uvs.dtype, device=_device)), dim=1
    )
    return rays


def project(points, intrinsics):
    """Project points in camera space to pixel coordinates based on intrinsics.
    Args:
        points (B, 3, H, W)
        intrinsics (B, 4): (fx, fy, cx, cy)

    Returns:
        pixels (B, 2, H, W)
    """
    rays = points / -points[:, -1:]
    # rays in pixel unit
    fxys = focal_length(intrinsics)
    uvs = rays[:, :2] * fxys.view(-1, 2, 1, 1)

    B, _, H, W = uvs.shape
    cs = principal_point(intrinsics, (H, W))
    # to pixels: (i, j) = (u, -v) + (cx, cy)
    uvs[:, 1] = -uvs[:, 1]  # flip v
    pixels = uvs + cs.view(-1, 2, 1, 1)
    return pixels


def pixels_to_points(intrinsics, depths, pixels):
    """Convert pixels to 3D points in camera space. (Camera facing -z direction)

    Args:
        intrinsics:
        depths (B, 1, H, W)
        pixels (B, 2, H, W)

    Returns:
        points (B, 3, H, W)

    """
    rays = pixels_to_rays(pixels, intrinsics)
    points = rays * depths
    return points


def reproject_points(points_cam_ref, extrinsics_ref, extrinsics_tgt):
    """Reproject points in reference camera coordinate to target camera coordinate

    Args:
        points_cam_ref (B, 3, H, W): points in reference camera coordinate.
        extrinsics_ref (B, 3, 4): [R, t] of reference camera.
        extrinsics_tgt (B, 3, 4): [R, t] of target_camera.

    Returns:
        points_cam_tgt (B, 3, H, W): points in target camera coordinate.

    """
    B, p_dim, H, W = points_cam_ref.shape
    assert p_dim == 3, "dimension of point {} != 3".format(p_dim)

    # t + R * p where t of (B, 3, 1), R of (B, 3, 3) and p of (B, 3, H*W)
    R_ref = extrinsics_ref[..., :p_dim]
    t_ref = extrinsics_ref[..., -1:]
    points_world = torch.baddbmm(t_ref, R_ref, points_cam_ref.view(B, p_dim, -1))

    # Reproject to target:
    # R'^T * (p - t') where t' of (B, 3, 1), R' of (B, 3, 3) and p of (B, 3, H*W)
    R_tgt = extrinsics_tgt[..., :p_dim]
    t_tgt = extrinsics_tgt[..., -1:]
    points_cam_tgt = torch.bmm(R_tgt.transpose(1, 2), points_world - t_tgt)
    return points_cam_tgt.view(B, p_dim, H, W)


def depth_to_points(depths, intrinsics):
    """
    Args:
        depths: (B, 1, H, W)
        intrinsics: (B, num_params)
    """
    B, _, H, W = depths.shape
    pixels = pixel_grid(B, (H, W))
    points_cam = pixels_to_points(intrinsics, depths, pixels)
    return points_cam


def calibrate_scale(extrinsics, intrinsics, depths):
    """Given depths, compute the global scale to adjust the extrinsics.
    Given a pair of depths, intrinsics, extrinsics, unproject the depth maps,
    rotate these points based on camera rotation and compute the center for each one.
    The distance between these centers should be of the same scale as the translation
    between the cameras. Therefore, let mu1, mu2 and t1, t2 be the two scene centers
    and the two camera projection centers. Then
        -scale * (t1 - t2) = mu1 - mu2.
    Therefore,
        scale = -dt.dot(dmu) / dt.dot(dt), where dt = t1 - t2, dmu = mu1 - mu2.

    Args:
        intrinsics (2, num_params)
        extrinsics (2, 3, 4): each one is [R, t]
        depths (2, 1, H, W)
    """
    assert (
        extrinsics.shape[0] == intrinsics.shape[0]
        and intrinsics.shape[0] == depths.shape[0]
    )
    points_cam = depth_to_points(depths, intrinsics)
    B, p_dim, H, W = points_cam.shape
    Rs = extrinsics[..., :p_dim]
    ts = extrinsics[..., p_dim]
    points_rot = torch.bmm(Rs, points_cam.view(B, p_dim, -1))
    mus = torch.mean(points_rot, axis=-1)
    # TODO(xuanluo): generalize this to more framse B>2 via variances of the points.
    assert B == 2
    dmu = mus[0] - mus[1]
    dt = ts[0] - ts[1]
    t_scale = -dt.dot(dmu) / dt.dot(dt)
    return t_scale


def warping_field(extrinsics, intrinsics, depths, tgt_ids: List[int]):
    """ Generate the warping field to warp the other frame the current frame.
    Args:
        intrinsics (N, num_params)
        extrinsics (N, 3, 4): each one is [R, t]
        depths (N, 1, H, W)
        tgt_ids (N, 1): warp frame tgt_ids[i] to i

    Returns:
        uvs (N, 2, H, W): sampling the other frame tgt_ids[i] with uvs[i] produces
            the current frame i.
    """
    assert (
        extrinsics.shape[0] == intrinsics.shape[0]
        and intrinsics.shape[0] == depths.shape[0]
    )

    points_cam = depth_to_points(depths, intrinsics)
    extrinsics_tgt = extrinsics[tgt_ids]
    points_tgt_cam = reproject_points(points_cam, extrinsics, extrinsics_tgt)
    uv_tgt = project(points_tgt_cam, intrinsics[tgt_ids])
    return uv_tgt


def sample(data, uv):
    """Sample data (B, C, H, W) by uv (B, 2, H, W) (in pixels). """
    H, W = data.shape[2:]
    # grid needs to be in [-1, 1] and (B, H, W, 2)
    # NOTE: divide by (W-1, H-1) instead of (W, H) because uv is in [-1,1]x[-1,1]
    size = torch.tensor((W - 1, H - 1), dtype=uv.dtype).view(1, -1, 1, 1).to(_device)
    grid = (2 * uv / size - 1).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(data, grid, padding_mode="border")


def warp_image(images, depths, extrinsics, intrinsics, tgt_ids: List[int]):
    """ Warp target images to the reference image based on depths and camera params
    Warp images[tgt_ids[i]] to images[i].

    Args:
        images (N, C, H, W)
        depths (N, 1, H, W)
        extrinsics (N, 3, 4)
        intrinsics (N, 4)
        tgt_ids (N, 1)

    Returns:
        images_warped
    """
    uv_tgt = warping_field(extrinsics, intrinsics, depths, tgt_ids)
    images_warped_to_ref = sample(images[tgt_ids], uv_tgt)
    return images_warped_to_ref
