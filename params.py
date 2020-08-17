#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse

from monodepth.depth_model_registry import get_depth_model, get_depth_model_list

from depth_fine_tuning import DepthFineTuningParams
from scale_calibration import ScaleCalibrationParams
from utils import frame_sampling, frame_range
from tools.colmap_processor import COLMAPParams
from tools.make_video import MakeVideoParams


class Video3dParamsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--op",
            choices=["all", "extract_frames"], default="all")

        self.parser.add_argument("--path", type=str,
            help="Path to all the input (except for the video) and output files "
            " are stored.")

        self.parser.add_argument("--video_file", type=str,
            help="Path to input video file. Will be ignored if `color_full` and "
            "`frames.txt` are already present.")

        self.parser.add_argument("--configure",
            choices=["default", "kitti"], default="default")

        self.add_video_args()
        self.add_flow_args()
        self.add_calibration_args()
        self.add_fine_tuning_args()
        self.add_make_video_args()

        self.initialized = True

    def add_video_args(self):
        self.parser.add_argument("--size", type=int, default=384,
            help="Size of the (long image dimension of the) output depth maps.")
        self.parser.add_argument("--align", type=int, default=0,
            help="Alignment requirement of the depth size (i.e, forcing each"
            " image dimension to be an integer multiple). If set <= 0 it will"
            " be set automatically, based on the requirements of the depth network.")

    def add_flow_args(self):
        self.parser.add_argument(
            "--flow_ops",
            nargs="*",
            help="optical flow operation: exhausted optical flow for all the pairs in"
            " dense_frame_range or consective that computes forward backward flow"
            " between consecutive frames.",
            choices=frame_sampling.SamplePairsMode.names(),
            default=["hierarchical2"],
        )
        self.parser.add_argument(
            "--flow_checkpoint", choices=["FlowNet2", "FlowNet2-KITTI"],
            default="FlowNet2"
        )
        self.parser.add_argument("--overlap_ratio", type=float, default=0.2)

    def add_calibration_args(self):
        COLMAPParams.add_arguments(self.parser)
        ScaleCalibrationParams.add_arguments(self.parser)

    def add_fine_tuning_args(self):
        DepthFineTuningParams.add_arguments(self.parser)
        self.parser.add_argument(
            "--model_type", type=str, choices=get_depth_model_list(),
            default="mc"
        )
        self.parser.add_argument(
            "--frame_range", default="",
            type=frame_range.parse_frame_range,
            help="Range of depth to fine-tune, e.g., 0,2-10,21-40."
        )

    def add_make_video_args(self):
        self.parser.add_argument("--make_video", action="store_true")
        MakeVideoParams.add_arguments(self.parser)

    def print(self):
        print("------------ Parameters -------------")
        args = vars(self.params)
        for k, v in sorted(args.items()):
            if type(v) == frame_range.NamedOptionalSet:
                print(f"{k}: '{v.name}'")
            else:
                print(f"{k}: {v}")
        print("-------------------------------------")

    def parse(self, args=None, namespace=None):
        if not self.initialized:
            self.initialize()
        self.params = self.parser.parse_args(args, namespace=namespace)

        if self.params.configure == "kitti":
            self.params.flow_checkpoint = "FlowNet2-KITTI"
            self.params.model_type = "monodepth2"
            self.params.overlap_ratio = 0.5
            if 'matcher' in self.params:
                self.params.matcher = 'sequential'

        # Resolve unspecified parameters
        model = get_depth_model(self.params.model_type)

        if self.params.align <= 0:
            self.params.align = model.align

        if self.params.learning_rate <= 0:
            self.params.learning_rate = model.learning_rate

        if self.params.lambda_view_baseline < 0:
            self.params.lambda_view_baseline = model.lambda_view_baseline

        self.print()

        return self.params
