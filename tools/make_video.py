#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import sys
import os
from os.path import join as pjoin
import shutil
import subprocess
from typing import Tuple, Optional, List

import cv2


LOG = logging.getLogger()
LOG.setLevel("INFO")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
LOG.handlers = []
LOG.addHandler(ch)


class MakeVideoParams:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "Create videos from color and depth frames. "
            "If <video3d_dir> is specified, <color_dir> and <depth_dirs> "
            "only need to be relative directory to <video3d_dir>."
            " <color_dir> and each of the <depth_dirs> should contain "
            "the same number of frames."
        )
        self.parser.add_argument(
            "--color_dir", default="color_down_png", help="directory of color images"
        )
        self.parser.add_argument(
            "--depth_dirs", nargs="*", help="directory of depth images"
        )
        self.parser.add_argument("--out_dir", help="output directory for the video")
        self.parser.add_argument("--ext", help="video extension", default=".mp4")
        self.parser.add_argument(
            "--frame_fmt", help="frame format", default="frame_%06d.png"
        )
        self.parser.add_argument(
            "--video3d_dir", help="directory for the 3D video", default=None
        )
        self.add_arguments(self.parser)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--ffmpeg", help="specify the path to the ffmpeg bin", default="ffmpeg"
        )


def parse_args():
    return MakeVideoParams().parser.parse_args()


def num_frames(dir, ext):
    return len([fn for fn in os.listdir(dir) if os.path.splitext(fn)[-1] == ext])


def augment_args(args):
    if args.video3d_dir is not None:
        args.color_dir = pjoin(args.video3d_dir, args.color_dir)
        args.depth_dirs = [pjoin(args.video3d_dir, dir) for dir in args.depth_dirs]
        args.out_dir = pjoin(args.video3d_dir, args.out_dir)

    # depth_dir can include or omit the "depth" suffix
    # number of frames should be equal
    frame_ext = os.path.splitext(args.frame_fmt)[-1]
    n = num_frames(args.color_dir, frame_ext)
    assert n > 0

    DEPTH = "depth"
    args.depth_names = []
    valid_depth_dirs = []
    for depth_dir in args.depth_dirs:
        names = os.listdir(depth_dir)
        if DEPTH in names and len(names) == 1:
            depth_dir = pjoin(depth_dir, DEPTH)

        depth_frames = num_frames(depth_dir, frame_ext)
        if depth_frames == n:
            valid_depth_dirs.append(depth_dir)
        else:
            logging.warning("[Warning] %d vs. %d in %s" % (depth_frames, n, depth_dir))
            continue

        p_head, p_tail = os.path.split(depth_dir)
        if p_tail == DEPTH:
            p_head, p_tail = os.path.split(p_head)
        args.depth_names.append(p_tail)
    args.depth_dirs = valid_depth_dirs
    return args


def frame_size(frame_fmt: str, frame_index: int = 0):
    im_fn = frame_fmt % frame_index
    return cv2.imread(im_fn).shape[:2][::-1]


def make_resized_filename(prefix: str, size: Tuple[int, int], ext: str):
    return prefix + ("_" + str(size)) + ext


def make_resized_filename_if_exists(
    prefix: str, ext: str, size: Optional[Tuple[int, int]] = None
) -> str:
    unsized_fn = prefix + ext
    if size is None:
        return unsized_fn
    sized_fn = make_resized_filename(prefix, size, ext)
    if os.path.isfile(sized_fn):
        return sized_fn
    return unsized_fn


def make_video(
    ffmpeg: str,
    frame_fmt: str,
    out_prefix: str,
    ext: str = ".mp4",
    size: Optional[Tuple[int, int]] = None,
    crf: int = 1,
) -> None:
    out_fn = out_prefix + ext
    if not os.path.isfile(out_fn):
        cmd = [
            ffmpeg,
            "-r", "30",
            "-i", frame_fmt,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            out_fn,
        ]
        print(subprocess.run(cmd, check=True))

    # resize the video if size is specified
    if size is None:
        return
    in_size = frame_size(frame_fmt)
    if in_size == size:
        return

    resized_out_fn = make_resized_filename(out_prefix, size, ext)
    if os.path.isfile(resized_out_fn):
        return

    resize_cmd = [
        ffmpeg,
        "-i",
        out_fn,
        "-vf",
        "scale=" + ":".join([str(x) for x in size]),
        resized_out_fn,
    ]
    print(subprocess.run(resize_cmd, check=True))


def make_overlay(depth_fmt: str, color_fmt: str, overlay_fmt: str) -> None:
    n = num_frames(os.path.dirname(color_fmt), os.path.splitext(color_fmt)[-1])
    for i in range(n):
        color = cv2.imread(color_fmt % i)
        depth = cv2.imread(depth_fmt % i)
        if depth.shape != color.shape:
            depth = cv2.resize(depth, color.shape[:2][::-1])
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        overlay = gray.reshape(gray.shape[:2] + (-1,)) / 2.0 + depth / 2.0
        cv2.imwrite(overlay_fmt % i, overlay)


def stack_videos(
    ffmpeg: str,
    fn_prefixes: List[str],
    out_dir: str,
    ext: str = ".mp4",
    size: Optional[Tuple[int, int]] = None,
    crf: int = 1,
) -> str:
    out_pre = "_".join([os.path.basename(pre) for pre in fn_prefixes])
    out_fn = pjoin(out_dir, out_pre + ext)
    if os.path.isfile(out_fn):
        return out_fn

    vid_fns = [
        make_resized_filename_if_exists(pre, ext, size=size) for pre in fn_prefixes
    ]
    cmd = [ffmpeg]
    for vid_fn in vid_fns:
        cmd.extend(["-i", vid_fn])
    cmd.extend(["-filter_complex", "hstack=inputs=" + str(len(vid_fns))])
    cmd.extend(["-crf", str(crf)])
    cmd.append(out_fn)
    print(subprocess.run(cmd, check=True))
    return out_fn


def make_depth_videos(
    ffmpeg: str,
    depth_fmt: str,
    color_fmt: str,
    out_prefix: str,
    ext: str = ".mp4",
    size: Optional[Tuple[int, int]] = None,
) -> None:
    # make a video using the depth frames
    make_video(ffmpeg, depth_fmt, out_prefix, ext=ext, size=size)

    # color depth overlay
    overlay_prefix = out_prefix + "-overlay"
    overlay_fn = overlay_prefix + ext
    if os.path.isfile(overlay_fn):
        return

    overlay_dir = out_prefix
    os.makedirs(overlay_dir, exist_ok=True)
    overlay_fmt = pjoin(overlay_dir, os.path.basename(depth_fmt))
    make_overlay(depth_fmt, color_fmt, overlay_fmt)
    make_video(ffmpeg, overlay_fmt, overlay_prefix, ext=ext, size=size)
    shutil.rmtree(overlay_dir)
    stack_videos(
        ffmpeg,
        [out_prefix, overlay_prefix],
        os.path.dirname(out_prefix),
        ext=ext,
        size=size,
    )


def main(args):
    COLOR_NAME = "color"

    args = augment_args(args)

    size = frame_size(pjoin(args.color_dir, args.frame_fmt))

    os.makedirs(args.out_dir, exist_ok=True)

    color_video_prefix = pjoin(args.out_dir, COLOR_NAME)
    make_video(
        args.ffmpeg,
        pjoin(args.color_dir, args.frame_fmt),
        color_video_prefix,
        ext=args.ext,
    )

    depth_video_prefixes = [pjoin(args.out_dir, name) for name in args.depth_names]
    for depth_dir, prefix in zip(args.depth_dirs, depth_video_prefixes):
        make_depth_videos(
            args.ffmpeg,
            pjoin(depth_dir, args.frame_fmt),
            pjoin(args.color_dir, args.frame_fmt),
            prefix,
            size=size,
            ext=args.ext,
        )
    if len(args.depth_dirs) > 0:
        stack_videos(
            args.ffmpeg,
            [color_video_prefix] + depth_video_prefixes,
            args.out_dir,
            size=size,
            ext=args.ext,
        )

        # merge overlay videos
        overlay_video_prefixes = []
        for pre in depth_video_prefixes:
            overlay_video_prefixes.extend([pre, pre + "-overlay"])
        stack_videos(
            args.ffmpeg, overlay_video_prefixes, args.out_dir, size=size, ext=args.ext
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
