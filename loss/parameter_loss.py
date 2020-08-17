#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch


class ParameterLoss(torch.nn.Module):
    def __init__(self, parameters_init, opt):
        self.parameters_init = parameters_init
        self.opt = opt
        assert opt.lambda_parameter > 0

    def __call__(self, parameters):
        sq_diff = [torch.abs(p - pi.data)
            for p, pi in zip(parameters, self.parameters_init)]

        sq_sum = torch.sum(torch.cat([d.flatten() for d in sq_diff]))
        loss = self.opt.lambda_parameter * sq_sum
        return loss, {"parameter_loss": loss.reshape(1, -1)}
