#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import cv2
from os.path import join as pjoin
import json
import math
import numpy as np
import torch.utils.data as data
import torch
from typing import Optional

from utils import image_io, frame_sampling as sampling


_dtype = torch.float32


def load_image(
    path: str,
    channels_first: bool,
    check_channels: Optional[int] = None,
    post_proc_raw=lambda x: x,
    post_proc_other=lambda x: x,
) -> torch.FloatTensor:
    if os.path.splitext(path)[-1] == ".raw":
        im = image_io.load_raw_float32_image(path)
        im = post_proc_raw(im)
    else:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        im = post_proc_other(im)
    im = im.reshape(im.shape[:2] + (-1,))

    if check_channels is not None:
        assert (
            im.shape[-1] == check_channels
        ), "receive image of shape {} whose #channels != {}".format(
            im.shape, check_channels
        )

    if channels_first:
        im = im.transpose((2, 0, 1))
    # to torch
    return torch.tensor(im, dtype=_dtype)


def load_color(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        torch.tensor. color in range [0, 1]
    """
    im = load_image(
        path,
        channels_first,
        post_proc_raw=lambda im: im[..., [2, 1, 0]] if im.ndim == 3 else im,
        post_proc_other=lambda im: im / 255,
    )
    return im


def load_flow(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        flow tensor in pixels.
    """
    flow = load_image(path, channels_first, check_channels=2)
    return flow


def load_mask(path: str, channels_first: bool) -> torch.ByteTensor:
    """
    Returns:
        mask takes value 0 or 1
    """
    mask = load_image(path, channels_first, check_channels=1) > 0
    return mask.to(_dtype)


class VideoDataset(data.Dataset):
    """Load 3D video frames and related metadata for optimizing consistency loss.
    File organization of the corresponding 3D video dataset should be
        color_down/frame_{__ID__:06d}.raw
        flow/flow_{__REF_ID__:06d}_{__TGT_ID__:06d}.raw
        mask/mask_{__REF_ID__:06d}_{__TGT_ID__:06d}.png
        metadata.npz: {'extrinsics': (N, 3, 4), 'intrinsics': (N, 4)}
        <flow_list.json>: [[i, j], ...]
    """

    def __init__(self, path: str, meta_file: str = None):
        """
        Args:
            path: folder path of the 3D video
        """
        self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.raw")
        if not os.path.isfile(self.color_fmt.format(0)):
            self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.png")

        self.mask_fmt = pjoin(path, "mask", "mask_{:06d}_{:06d}.png")
        self.flow_fmt = pjoin(path, "flow", "flow_{:06d}_{:06d}.raw")

        if meta_file is not None:
            with open(meta_file, "rb") as f:
                meta = np.load(f)
                self.extrinsics = torch.tensor(meta["extrinsics"], dtype=_dtype)
                self.intrinsics = torch.tensor(meta["intrinsics"], dtype=_dtype)
            assert (
                self.extrinsics.shape[0] == self.intrinsics.shape[0]
            ), "#extrinsics({}) != #intrinsics({})".format(
                self.extrinsics.shape[0], self.intrinsics.shape[0]
            )

        flow_list_fn = pjoin(path, "flow_list.json")
        if os.path.isfile(flow_list_fn):
            with open(flow_list_fn, "r") as f:
                self.flow_indices = json.load(f)
        else:
            names = os.listdir(os.path.dirname(self.flow_fmt))
            self.flow_indices = [
                self.parse_index_pair(name)
                for name in names
                if os.path.splitext(name)[-1] == os.path.splitext(self.flow_fmt)[-1]
            ]
            self.flow_indices = sampling.to_in_range(self.flow_indices)
        self.flow_indices = list(sampling.SamplePairs.to_one_way(self.flow_indices))

    def parse_index_pair(self, name):
        strs = os.path.splitext(name)[0].split("_")[-2:]
        return [int(s) for s in strs]

    def __getitem__(self, index: int):
        """Fetch tuples of data. index = i * (i-1) / 2 + j, where i > j for pair (i,j)
        So [-1+sqrt(1+8k)]/2 < i <= [1+sqrt(1+8k))]/2, where k=index. So
            i = floor([1+sqrt(1+8k))]/2)
            j = k - i * (i - 1) / 2.

        The number of image frames fetched, N, is not the 1, but computed
        based on what kind of consistency to be measured.
        For instance, geometry_consistency_loss requires random pairs as samples.
        So N = 2.
        If with more losses, say triplet one from temporal_consistency_loss. Then
            N = 2 + 3.

        Returns:
            stacked_images (N, C, H, W): image frames
            targets: {
                'extrinsics': torch.tensor (N, 3, 4), # extrinsics of each frame.
                                Each (3, 4) = [R, t].
                                    point_wolrd = R * point_cam + t
                'intrinsics': torch.tensor (N, 4), # (fx, fy, cx, cy) for each frame
                'geometry_consistency':
                    {
                        'indices':  torch.tensor (2),
                                    indices for corresponding pairs
                                        [(ref_index, tgt_index), ...]
                        'flows':    ((2, H, W),) * 2 in pixels.
                                    For k in range(2) (ref or tgt),
                                        pixel p = pixels[indices[b, k]][:, i, j]
                                    correspond to
                                        p + flows[k][b, :, i, j]
                                    in frame indices[b, (k + 1) % 2].
                        'masks':    ((1, H, W),) * 2. Masks of valid flow matches
                                    to compute the consistency in training.
                                    Values are 0 or 1.
                    }
            }

        """
        pair = self.flow_indices[index]

        indices = torch.tensor(pair)
        intrinsics = torch.stack([self.intrinsics[k] for k in pair], dim=0)
        extrinsics = torch.stack([self.extrinsics[k] for k in pair], dim=0)

        images = torch.stack(
            [load_color(self.color_fmt.format(k), channels_first=True) for k in pair],
            dim=0,
        )
        flows = [
            load_flow(self.flow_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]
        masks = [
            load_mask(self.mask_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]

        metadata = {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "geometry_consistency": {
                "indices": indices,
                "flows": flows,
                "masks": masks,
            },
        }

        if getattr(self, "scales", None):
            if isinstance(self.scales, dict):
                metadata["scales"] = torch.stack(
                    [torch.Tensor([self.scales[k]]) for k in pair], dim=0
                )
            else:
                metadata["scales"] = torch.Tensor(
                    [self.scales, self.scales]).reshape(2, 1)

        return (images, metadata)

    def __len__(self):
        return len(self.flow_indices)


class VideoFrameDataset(data.Dataset):
    """Load video frames from
        color_fmt.format(frame_id)
    """

    def __init__(self, color_fmt, frames=None):
        """
        Args:
            color_fmt: e.g., <video_dir>/frame_{:06d}.raw
        """
        self.color_fmt = color_fmt

        if frames is None:
            files = os.listdir(os.path.dirname(self.color_fmt))
            self.frames = range(len(files))
        else:
            self.frames = frames

    def __getitem__(self, index):
        """Fetch image frame.
        Returns:
            image (C, H, W): image frames
        """
        frame_id = self.frames[index]
        image = load_color(self.color_fmt.format(frame_id), channels_first=True)
        meta = {"frame_id": frame_id}
        return image, meta

    def __len__(self):
        return len(self.frames)
