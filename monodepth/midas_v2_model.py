#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch

from utils.url_helpers import get_model_from_url

from .midas_v2.midas_net import MidasNet
from .depth_model import DepthModel


class MidasV2Model(DepthModel):
    # Requirements and default settings
    align = 32
    learning_rate = 0.0001
    lambda_view_baseline = 0.0001

    def __init__(self, support_cpu: bool = False, pretrained: bool = True):
        super().__init__()

        if support_cpu:
            # Allow the model to run on CPU when GPU is not available.
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Rather raise an error when GPU is not available.
            self.device = torch.device("cuda")

        self.model = MidasNet(non_negative=True)

        # Load pretrained checkpoint
        if pretrained:
            checkpoint = (
                "https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt"
            )
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint, progress=True, check_hash=True
            )
            self.model.load_state_dict(state_dict)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        self.norm_mean = torch.Tensor(
            [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.norm_stdev = torch.Tensor(
            [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    def estimate_depth(self, images):
        # Reshape ...CHW -> XCHW
        shape = images.shape
        C, H, W = shape[-3:]
        input_ = images.reshape(-1, C, H, W).to(self.device)

        input_ = (input_ - self.norm_mean.to(self.device)) / \
            self.norm_stdev.to(self.device)

        output = self.model(input_)

        # Reshape X1HW -> BNHW
        depth = output.reshape(shape[:-3] + output.shape[-2:])

        # Convert from disparity to depth
        depth = depth.reciprocal()

        return depth

    def save(self, file_name):
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_name)
