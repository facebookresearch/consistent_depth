#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import numpy as np
import os
from os.path import join as pjoin
import logging
from typing import Optional, Set
import torch

from utils.helpers import SuppressedStdout

from loaders.video_dataset import _dtype, load_color
from tools.colmap_processor import COLMAPParams, COLMAPProcessor
from utils import (
    image_io,
    geometry,
    load_colmap,
    visualization,
)
from utils.helpers import print_banner
from utils.torch_helpers import _device


class ScaleCalibrationParams:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--dense_frame_ratio", type=float, default=0.95,
            help="threshold on percentage of successully computed dense depth frames."
        )
        parser.add_argument("--dense_pixel_ratio", type=float, default=0.3,
            help="ratio of valid dense depth pixels for that frame to valid")


def prepare_colmap_color(video):
    """
        If there is no dynamic object mask (in `mask_dynamic`) then just
        use `color_full` to do colmap so return `color_full`. Otherwise, set
        dynamic part to be black. `mask_dynamic` is 1 in static part
        and 0 in dynamic part. So in this case, return 'color_colmap_dense'

        Returns:
            output_directory
    """
    print('Preparint color input for COLMAP...')

    out_dir = pjoin(video.path, 'color_colmap_dense')
    dynamic_mask_dir = pjoin(video.path, 'mask_dynamic')
    color_src_dir = pjoin(video.path, 'color_full')
    if not os.path.isdir(dynamic_mask_dir):
        return color_src_dir

    if video.check_frames(out_dir, 'png'):
        return out_dir

    name_fmt = 'frame_{:06d}.png'
    os.makedirs(out_dir, exist_ok=True)
    for i in range(video.frame_count):
        name = name_fmt.format(i)
        im = cv2.imread(pjoin(color_src_dir, name))
        seg_fn = pjoin(dynamic_mask_dir, name)
        seg = (cv2.imread(seg_fn, 0) > 0)[..., np.newaxis]
        masked = im * seg
        cv2.imwrite(pjoin(out_dir, name), masked)

    assert video.check_frames(out_dir, 'png')

    return out_dir


def make_camera_params_from_colmap(path, sparse_dir):
    cameras, images, points3D = load_colmap.read_model(path=sparse_dir, ext=".bin")
    size_new = image_io.load_raw_float32_image(
        pjoin(path, "color_down", "frame_{:06d}.raw".format(0))
    ).shape[:2][::-1]
    intrinsics, extrinsics = load_colmap.convert_calibration(
        cameras, images, size_new
    )
    return intrinsics, extrinsics


def visualize_calibration_pair(
    extrinsics, intrinsics, depth_fmt, color_fmt, id_pair, vis_dir
):
    assert len(id_pair) == 2

    depth_fns = [depth_fmt.format(id) for id in id_pair]
    if any(not os.path.isfile(fn) for fn in depth_fns):
        return

    color_fns = [color_fmt.format(id) for id in id_pair]
    colors = [load_color(fn, channels_first=True) for fn in color_fns]
    colors = torch.stack(colors, dim=0).to(_device)
    inv_depths = [image_io.load_raw_float32_image(fn) for fn in depth_fns]
    depths = 1.0 / torch.tensor(inv_depths, device=_device).unsqueeze(-3)

    def select_tensor(x):
        return torch.tensor(x[list(id_pair)], device=_device, dtype=_dtype)

    extr = select_tensor(extrinsics)
    intr = select_tensor(intrinsics)

    colors_warped_to_ref = geometry.warp_image(colors, depths, extr, intr, [1, 0])

    def vis(x):
        x = np.clip(x.permute(1, 2, 0).cpu().numpy(), a_min=0, a_max=1)
        x = x[..., ::-1] * 255  # RGB to BGR, [0, 1] to [0, 255]
        return x

    os.makedirs(vis_dir, exist_ok=True)
    for id, tgt_id, color_warped, color in zip(
        id_pair, id_pair[::-1], colors_warped_to_ref, colors
    ):
        cv2.imwrite(pjoin(vis_dir, "frame_{:06d}.png".format(id)), vis(color))
        cv2.imwrite(
            pjoin(vis_dir, "frame_{:06d}_warped_to_{:06d}.png".format(tgt_id, id)),
            vis(color_warped),
        )


def visualize_all_calibration(
    extrinsics, intrinsics, depth_fmt, color_fmt, frame_range, vis_dir
):
    id_pairs = [
        (frame_range.index_to_frame[i], frame_range.index_to_frame[0])
        for i in range(1, len(frame_range))
    ]

    for id_pair in id_pairs:
        visualize_calibration_pair(
            extrinsics, intrinsics, depth_fmt, color_fmt, id_pair, vis_dir
        )


def check_frames(
    src_dir, src_ext, dst_dir, dst_ext,
    frame_names: Optional[Set[str]] = None
):
    if not os.path.isdir(src_dir):
        assert frame_names is not None
        names = list(frame_names)
    else:
        names = [n.replace(src_ext, dst_ext)
            for n in os.listdir(src_dir) if n.endswith(src_ext)]
        names = [n for n in names if frame_names is None or n in frame_names]
    return all(
        os.path.isfile(pjoin(dst_dir, n))
        for n in names
    )


def calibrate_scale(video, out_dir, frame_range, args):
    # COLMAP reconstruction.
    print_banner("COLMAP reconstruction")

    colmap_dir = pjoin(video.path, 'colmap_dense')
    src_meta_file = pjoin(colmap_dir, "metadata.npz")

    colmap = COLMAPProcessor(args.colmap_bin_path)
    dense_dir = colmap.dense_dir(colmap_dir, 0)

    if os.path.isfile(src_meta_file):
        print("Checked metadata file exists.")
    else:
        color_dir = prepare_colmap_color(video)

        if not colmap.check_dense(
            dense_dir, color_dir, valid_ratio=args.dense_frame_ratio
        ):
            path_args = [color_dir, colmap_dir]
            mask_path = pjoin(video.path, 'colmap_mask')
            if os.path.isdir(mask_path):
                path_args.extend(['--mask_path', mask_path])
            colmap_args = COLMAPParams().parse_args(
                args=path_args + ['--dense_max_size', str(args.size)],
                namespace=args
            )

            colmap.process(colmap_args)

        intrinsics, extrinsics = make_camera_params_from_colmap(
            video.path, colmap.sparse_dir(colmap_dir, 0)
        )
        np.savez(src_meta_file, intrinsics=intrinsics, extrinsics=extrinsics)

    # Convert COLMAP dense depth maps to .raw file format.
    print_banner("Convert COLMAP depth maps")

    converted_depth_fmt = pjoin(
        video.path, "depth_colmap_dense", "depth", "frame_{:06d}.raw"
    )

    # convert colmap dense depths to .raw
    converted_depth_dir = os.path.dirname(converted_depth_fmt)
    dense_depth_dir = pjoin(dense_dir, "stereo", "depth_maps")
    frames = frame_range.frames()
    if not check_frames(
        dense_depth_dir, colmap.dense_depth_suffix(), converted_depth_dir, "",
        frame_names={f"frame_{i:06d}.png" for i in frames},
    ):
        os.makedirs(converted_depth_dir, exist_ok=True)
        colmap_depth_fmt = pjoin(
            dense_depth_dir, "frame_{:06d}.png" + colmap.dense_depth_suffix()
        )
        for i in frames:
            colmap_depth_fn = colmap_depth_fmt.format(i)
            if not os.path.isfile(colmap_depth_fn):
                logging.warning(
                    "[SCALE CALIBRATION] %s does not exist.",
                    colmap_depth_fn
                )
                continue
            cmp_depth = load_colmap.read_array(colmap_depth_fn)
            inv_cmp_depth = 1.0 / cmp_depth
            ix = np.isinf(inv_cmp_depth) | (inv_cmp_depth < 0)
            inv_cmp_depth[ix] = float("nan")
            image_io.save_raw_float32_image(
                converted_depth_fmt.format(i), inv_cmp_depth
            )
        with SuppressedStdout():
            visualization.visualize_depth_dir(
                converted_depth_dir, converted_depth_dir,
                force=True, min_percentile=0, max_percentile=99,
            )

    # Compute scaled depth maps
    print_banner("Compute per-frame scales")

    scaled_depth_dir = pjoin(out_dir, "depth_scaled_by_colmap_dense", "depth")
    scaled_depth_fmt = pjoin(scaled_depth_dir, "frame_{:06d}.raw")
    scales_file = pjoin(out_dir, "scales.csv")
    src_depth_fmt = pjoin(
        video.path, f"depth_{args.model_type}", "depth", "frame_{:06d}.raw"
    )
    frames = frame_range.frames()

    if (
        check_frames(
            converted_depth_dir, ".png",
            os.path.dirname(scaled_depth_fmt), ".raw"
        )
        and os.path.isfile(scales_file)
    ):
        src_to_colmap_scales = np.loadtxt(scales_file, delimiter=',')
        assert src_to_colmap_scales.shape[0] >= len(frames) * args.dense_frame_ratio \
            and src_to_colmap_scales.shape[1] == 2, \
            (f"scales shape is {src_to_colmap_scales.shape} does not match "
             + f"({len(frames)}, 2) with threshold {args.dense_frame_ratio}")
        print("Existing scales file loaded.")
    else:
        # Scale depth maps
        os.makedirs(scaled_depth_dir, exist_ok=True)
        src_to_colmap_scales_map = {}

        for i in frames:
            converted_depth_fn = converted_depth_fmt.format(i)
            if not os.path.isfile(converted_depth_fn):
                logging.warning("[SCALE CALIBRATION] %s does not exist",
                    converted_depth_fn)
                continue
            # convert colmap_depth to raw
            inv_cmp_depth = image_io.load_raw_float32_image(converted_depth_fn)
            # compute scale for init depths
            inv_src_depth = image_io.load_raw_float32_image(src_depth_fmt.format(i))
            # src_depth * scale = (1/inv_src_depth) * scale == cmp_depth
            inv_cmp_depth = cv2.resize(
                inv_cmp_depth, inv_src_depth.shape[:2][::-1],
                interpolation=cv2.INTER_NEAREST
            )
            ix = np.isfinite(inv_cmp_depth)

            if np.sum(ix) / ix.size < args.dense_pixel_ratio:
                # not enough pixels are valid and hence the frame is invalid.
                continue

            scales = (inv_src_depth / inv_cmp_depth)[ix]
            scale = np.median(scales)
            print(f"Scale[{i}]: median={scale}, std={np.std(scales)}")
            # scale = np.median(inv_depth) * np.median(cmp_depth)
            src_to_colmap_scales_map[i] = float(scale)
            scaled_inv_src_depth = inv_src_depth / scale
            image_io.save_raw_float32_image(
                scaled_depth_fmt.format(i), scaled_inv_src_depth
            )
        with SuppressedStdout():
            visualization.visualize_depth_dir(
                scaled_depth_dir, scaled_depth_dir, force=True
            )

        # Write scales.csv
        xs = sorted(src_to_colmap_scales_map.keys())
        ys = [src_to_colmap_scales_map[x] for x in xs]
        src_to_colmap_scales = np.stack((np.array(xs), np.array(ys)), axis=-1)
        np.savetxt(scales_file, src_to_colmap_scales, delimiter=",")

    valid_frames = {int(s) for s in src_to_colmap_scales[:, 0]}

    # Scale the extrinsics' translations
    scaled_meta_file = pjoin(out_dir, "metadata_scaled.npz")
    if os.path.isfile(scaled_meta_file):
        print("Scaled metadata file exists.")
    else:
        scales = src_to_colmap_scales[:, 1]
        mean_scale = scales.mean()
        print(f"[scales] mean={mean_scale}, std={np.std(scales)}")

        with np.load(src_meta_file) as meta_colmap:
            intrinsics = meta_colmap["intrinsics"]
            extrinsics = meta_colmap["extrinsics"]

        extrinsics[..., -1] /= mean_scale
        np.savez(
            scaled_meta_file,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            scales=src_to_colmap_scales,
        )

        color_fmt = pjoin(video.path, "color_down", "frame_{:06d}.raw")
        vis_dir = pjoin(out_dir, "vis_calibration_dense")
        visualize_all_calibration(
            extrinsics, intrinsics, scaled_depth_fmt,
            color_fmt, frame_range, vis_dir,
        )

    return valid_frames
