#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch


_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(data):
    if isinstance(data, torch.Tensor):
        data = data.to(_device, non_blocking=True)
        return data

    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = to_device(v)
        return data

    # list or tuple
    for i, v in enumerate(data):
        data[i] = to_device(v)
    return data
