#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.


class LossParams:
    """
    Loss related parameters
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--lambda_view_baseline",
            type=float,
            default=-1,
            help="The baseline to define weight to penalize disparity difference."
            " If < 0 it will be set automatically to the default for the"
            " specified model adapter.",
        )
        parser.add_argument(
            "--lambda_reprojection",
            type=float,
            default=1.0,
            help="weight for reprojection loss.",
        )
        parser.add_argument(
            "--lambda_parameter",
            type=float,
            default=0,
            help="weight for network parameter regularization loss.",
        )
        return parser

    @staticmethod
    def make_str(opt):
        return (
            "B{}".format(opt.lambda_view_baseline)
            + "_R{}".format(opt.lambda_reprojection)
            + '_PL1-{}'.format(opt.lambda_parameter)
        )
