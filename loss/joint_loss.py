#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

import torch
from torch.nn import Parameter

from .parameter_loss import ParameterLoss
from .consistency_loss import ConsistencyLoss
from utils.torch_helpers import _device
from loaders.video_dataset import _dtype


class JointLoss(torch.nn.Module):
    def __init__(self, opt, parameters_init=None):
        super().__init__()
        self.opt = opt
        if opt.lambda_parameter > 0:
            assert parameters_init is not None
            self.parameter_loss = ParameterLoss(parameters_init, opt)

        if opt.lambda_view_baseline > 0 or opt.lambda_reprojection > 0:
            self.consistency_loss = ConsistencyLoss(opt)

    def __call__(
        self,
        depths,
        metadata,
        parameters: Optional[List[Parameter]] = None,
    ):
        loss = torch.zeros(1, dtype=_dtype, device=_device)
        batch_losses = {}
        if self.opt.lambda_parameter > 0:
            assert parameters is not None
            para_loss, para_batch_losses = self.parameter_loss(parameters)
            loss += para_loss
            batch_losses.update(para_batch_losses)

        if self.opt.lambda_view_baseline > 0 or self.opt.lambda_reprojection > 0:
            consis_loss, consis_batch_losses = self.consistency_loss(
                depths, metadata,
            )
            loss += consis_loss
            batch_losses.update(consis_batch_losses)

        return loss, batch_losses
