#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch.nn


def sample(data, uv):
    """Sample data (H, W, <C>) by uv (H, W, 2) (in pixels). """
    shape = data.shape
    # data from (H, W, <C>) to (1, C, H, W)
    data = data.reshape(data.shape[:2] + (-1,))
    data = torch.tensor(data).permute(2, 0, 1)[None, ...]
    # (H, W, 2) -> (1, H, W, 2)
    uv = torch.tensor(uv)[None, ...]

    H, W = shape[:2]
    # grid needs to be in [-1, 1] and (B, H, W, 2)
    size = torch.tensor((W, H), dtype=uv.dtype).view(1, 1, 1, -1)
    grid = (2 * uv / size - 1).to(data.dtype)
    tensor = torch.nn.functional.grid_sample(data, grid, padding_mode="border")
    # from (1, C, H, W) to (H, W, <C>)
    return tensor.permute(0, 2, 3, 1).reshape(shape).numpy()


def sse(x, y, axis=-1):
    """Sum of suqare error"""
    d = x - y
    return np.sum(d * d, axis=axis)


def consistency_mask(im_ref, im_tgt, flow, threshold, diff_func=sse):
    H, W = im_ref.shape[:2]
    im_ref = im_ref.reshape(H, W, -1)
    im_tgt = im_tgt.reshape(H, W, -1)
    x, y = np.arange(W), np.arange(H)
    X, Y = np.meshgrid(x, y)
    u, v = flow[..., 0], flow[..., 1]
    idx_x, idx_y = u + X, v + Y

    # first constrain to within the image
    mask = np.all(
        np.stack((idx_x >= 0, idx_x <= W - 1, 0 <= idx_y, idx_y <= H - 1), axis=-1),
        axis=-1,
    )

    im_tgt_to_ref = sample(im_tgt, np.stack((idx_x, idx_y), axis=-1))

    mask = np.logical_and(mask, diff_func(im_ref, im_tgt_to_ref) < threshold)
    return mask


def consistent_flow_masks(flows, colors, flow_thresh, color_thresh):
    # mask from flow consistency
    masks_flow = [
        consistency_mask(flow_ref, -flow_tgt, flow_ref, flow_thresh ** 2)
        for flow_ref, flow_tgt in zip(flows, flows[::-1])
    ]
    # mask from photometric consistency
    C = colors[0].shape[-1]
    masks_photo = [
        consistency_mask(c_ref, c_tgt, flow_ref, C * (color_thresh ** 2))
        for c_ref, c_tgt, flow_ref in zip(colors, colors[::-1], flows)
    ]
    # merge the two
    masks = [np.logical_and(mf, mp) for mf, mp in zip(masks_flow, masks_photo)]
    return masks
