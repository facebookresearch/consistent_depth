#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
from os.path import join as pjoin
import shutil

from depth_fine_tuning import DepthFineTuner
from flow import Flow
from scale_calibration import calibrate_scale
from tools import make_video as mkvid
from utils.frame_range import FrameRange, OptionalSet
from utils.helpers import print_banner, print_title
from video import (Video, sample_pairs)


class DatasetProcessor:
    def __init__(self, writer=None):
        self.writer = writer

    def create_output_path(self, params):
        range_tag = f"R{params.frame_range.name}"
        flow_ops_tag = "-".join(params.flow_ops)
        name = f"{range_tag}_{flow_ops_tag}_{params.model_type}"

        out_dir = pjoin(self.path, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def extract_frames(self, params):
        print_banner("Extracting PTS")
        self.video.extract_pts()

        print_banner("Extracting frames")
        self.video.extract_frames()

    def pipeline(self, params):
        self.extract_frames(params)

        print_banner("Downscaling frames (raw)")
        self.video.downscale_frames("color_down", params.size, "raw")

        print_banner("Downscaling frames (png)")
        self.video.downscale_frames("color_down_png", params.size, "png")

        print_banner("Downscaling frames (for flow)")
        self.video.downscale_frames("color_flow", Flow.max_size(), "png", align=64)

        frame_range = FrameRange(
            frame_range=params.frame_range.set, num_frames=self.video.frame_count,
        )
        frames = frame_range.frames()

        print_banner("Compute initial depth")

        ft = DepthFineTuner(self.out_dir, frames, params)
        initial_depth_dir = pjoin(self.path, f"depth_{params.model_type}")
        if not self.video.check_frames(pjoin(initial_depth_dir, "depth"), "raw"):
            ft.save_depth(initial_depth_dir)

        valid_frames = calibrate_scale(self.video, self.out_dir, frame_range, params)
        # frame range for finetuning:
        ft_frame_range = frame_range.intersection(OptionalSet(set(valid_frames)))
        print("Filtered out frames",
            sorted(set(frame_range.frames()) - set(ft_frame_range.frames())))

        print_banner("Compute flow")

        frame_pairs = sample_pairs(ft_frame_range, params.flow_ops)
        self.flow.compute_flow(frame_pairs, params.flow_checkpoint)

        print_banner("Compute flow masks")

        self.flow.mask_valid_correspondences()

        flow_list_path = self.flow.check_good_flow_pairs(
            frame_pairs, params.overlap_ratio
        )
        shutil.copyfile(flow_list_path, pjoin(self.path, "flow_list.json"))

        print_banner("Visualize flow")

        self.flow.visualize_flow(warp=True)

        print_banner("Fine-tuning")

        ft.fine_tune(writer=self.writer)

        print_banner("Compute final depth")

        if not self.video.check_frames(pjoin(ft.out_dir, "depth"), "raw", frames):
            ft.save_depth(ft.out_dir, frames)

        if params.make_video:
            print_banner("Export visualization videos")
            self.make_videos(params, ft.out_dir)

        return initial_depth_dir, ft.out_dir, frame_range.frames()

    def process(self, params):
        self.path = params.path
        os.makedirs(self.path, exist_ok=True)

        self.video_file = params.video_file

        self.out_dir = self.create_output_path(params)

        self.video = Video(params.path, params.video_file)
        self.flow = Flow(params.path, self.out_dir)

        print_title(f"Processing dataset '{self.path}'")

        print(f"Output directory: {self.out_dir}")

        if params.op == "all":
            return self.pipeline(params)
        elif params.op == "extract_frames":
            return self.extract_frames(params)
        else:
            raise RuntimeError("Invalid operation specified.")

    def make_videos(self, params, ft_depth_dir):
        args = [
            "--color_dir", pjoin(self.path, "color_down_png"),
            "--out_dir", pjoin(self.out_dir, "videos"),
            "--depth_dirs",
            pjoin(self.path, f"depth_{params.model_type}"),
            pjoin(self.path, "depth_colmap_dense"),
            pjoin(ft_depth_dir, "depth"),
        ]
        gt_dir = pjoin(self.path, "depth_gt")
        if os.path.isdir(gt_dir):
            args.append(gt_dir)

        vid_params = mkvid.MakeVideoParams().parser.parse_args(
            args,
            namespace=params
        )
        logging.info("Make videos {}".format(vid_params))
        mkvid.main(vid_params)
