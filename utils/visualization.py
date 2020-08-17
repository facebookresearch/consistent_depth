#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import numpy
import os
import subprocess
import sys
import logging

from matplotlib.cm import get_cmap

from . import image_io


CM_MAGMA = (numpy.array([get_cmap('magma').colors]).
            transpose([1, 0, 2]) * 255)[..., ::-1].astype(numpy.uint8)


def visualize_depth(depth, depth_min=None, depth_max=None):
    """Visualize the depth map with colormap.

    Rescales the values so that depth_min and depth_max map to 0 and 1,
    respectively.
    """
    if depth_min is None:
        depth_min = numpy.amin(depth)

    if depth_max is None:
        depth_max = numpy.amax(depth)

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled = depth_scaled ** 0.5
    depth_scaled_uint8 = numpy.uint8(depth_scaled * 255)

    return ((cv2.applyColorMap(
        depth_scaled_uint8, CM_MAGMA) / 255) ** 2.2) * 255


def visualize_depth_dir(
    src_dir: str, dst_dir: str, force: bool = False, extension: str = ".raw",
    min_percentile: float = 0, max_percentile: float = 100,
):
    src_files = []
    dst_files = []
    for file in sorted(os.listdir(src_dir)):
        base, ext = os.path.splitext(file)
        if ext.lower() == extension:
            src_files.append(file)
            dst_files.append(f"{base}.png")

    if len(src_files) == 0:
        return

    # Check if all dst_files already exist
    dst_exists = True
    for file in dst_files:
        if not os.path.exists(f"{dst_dir}/{file}"):
            dst_exists = False
            break

    if not force and dst_exists:
        return

    d_min = sys.float_info.max
    d_max = sys.float_info.min

    for src_file in src_files:
        print("reading '%s'." % src_file)
        if extension == ".raw":
            disparity = image_io.load_raw_float32_image(f"{src_dir}/{src_file}")
        else:
            disparity = cv2.imread(f"{src_dir}/{src_file}")
        ix = numpy.isfinite(disparity)

        if numpy.sum(ix) == 0:
            logging.warning(f"{src_file} has 0 valid depth")
            continue

        valid_disp = disparity[ix]
        d_min = min(d_min, numpy.percentile(valid_disp, min_percentile))
        d_max = max(d_max, numpy.percentile(valid_disp, max_percentile))

    for i in range(len(src_files)):
        src_file = src_files[i]
        dst_file = dst_files[i]

        print(f"reading '{src_file}'.")
        if os.path.exists(f"{dst_dir}/{dst_file}") and not force:
            print(f"skipping existing file '{dst_file}'.")
        else:
            if extension == ".raw":
                disparity = image_io.load_raw_float32_image(
                    f"{src_dir}/{src_file}")
            else:
                disparity = cv2.imread(f"{src_dir}/{src_file}")

            disparity_vis = visualize_depth(disparity, d_min, d_max)

            print(f"writing '{dst_file}'.")
            cv2.imwrite(f"{dst_dir}/{dst_file}", disparity_vis)


def create_video(pattern: str, output_file: str, ffmpeg_bin: str = 'ffmpeg'):
    if not os.path.exists(output_file):
        cmd = [ffmpeg_bin, "-r", "30",
            "-i", pattern,
            "-c:v", "libx264",
            "-crf", "27",
            "-pix_fmt", "yuv420p",
            output_file]
        subprocess.call(cmd)


def apply_mask(im, mask, mask_color=None):
    im = im.reshape(im.shape[:2] + (-1,))
    C = im.shape[-1]
    mask = mask.reshape(mask.shape[:2] + (-1,)) > 0
    if mask_color is None:
        mask_color = numpy.array([0, 255, 0] if C == 3 else 1)
    mask_color = mask_color.reshape(1, 1, C)
    inv_mask = (1 - mask) * mask_color
    result = 0.7 * im + 0.3 * inv_mask
    return result.squeeze()
