#!/usr/bin/env python3
from torch.optim.optimizer import Optimizer
from torch.optim import Adam

OPTIMIZER_MAP = {
    "Adam": Adam,
}


OPTIMIZER_NAMES = OPTIMIZER_MAP.keys()


OPTIMIZER_CLASSES = OPTIMIZER_MAP.values()


def create(optimizer_name: str, *args, **kwargs) -> Optimizer:
    return OPTIMIZER_MAP[optimizer_name](*args, **kwargs)
