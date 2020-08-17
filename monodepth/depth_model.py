#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from abc import abstractmethod
import torch


class DepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, metadata=None):
        """
        Images should be feed in the format (N, C, H, W). Channels are in BGR
        order and values in [0, 1].

        Metadata is not used by the depth models itself, only here, for value
        transformations.

        metadata["scales"]: (optional, can be None) specifies a post-scale
            transformation of the depth values. Format (1, N, 1).
        """
        depth = self.estimate_depth(images)

        if metadata is not None:
            if "scales" in metadata:
                factor = metadata["scales"].unsqueeze(3).cuda()
                depth = depth * factor

        return depth

    @abstractmethod
    def estimate_depth(self, images, metadata) -> torch.Tensor:
        pass

    @abstractmethod
    def save(self, label):
        pass
