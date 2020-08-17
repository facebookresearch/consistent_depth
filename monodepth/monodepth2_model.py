#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from os.path import join as pjoin

import torch

from utils.url_helpers import get_model_from_url

from .depth_model import DepthModel
from .monodepth2.networks.resnet_encoder import ResnetEncoder
from .monodepth2.networks.depth_decoder import DepthDecoder


class Monodepth2Model(DepthModel):
    # Requirements and default settings
    align = 1
    learning_rate = 0.00004
    lambda_view_baseline = 1

    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda")

        model_url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono+stereo_1024x320.zip"
        local_model_dir = get_model_from_url(model_url, "monodepth2_mono+stereo_1024x320/", is_zip=True)
        encoder_model_file = pjoin(local_model_dir, "encoder.pth")
        depth_model_file = pjoin(local_model_dir, "depth.pth")

        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_model_file, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        print(f"Model was trained at {self.feed_width} x {self.feed_height}.")
        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)

        self.depth_decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_model_file, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)

    def train(self):
        self.encoder.train()
        self.depth_decoder.train()

    def eval(self):
        self.encoder.eval()
        self.depth_decoder.eval()

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.depth_decoder.parameters())

    def estimate_depth(self, images):
        # import pdb; pdb.set_trace()

        # Reshape ...CHW -> NCHW
        shape = images.shape
        C, H, W = shape[-3:]
        images = images.reshape(-1, C, H, W)

        # Estimate depth
        feed_size = [self.feed_height, self.feed_width]
        images = torch.nn.functional.interpolate(
            images, size=feed_size, mode='bicubic', align_corners=False)

        features = self.encoder(images)
        outputs = self.depth_decoder(features)

        disparity = outputs[("disp", 0)]
        disparity = torch.nn.functional.interpolate(
            disparity, size=[H, W], mode='bicubic', align_corners=False)

        depth = disparity.reciprocal()

        # Reshape N1HW -> ...1HW
        out_shape = shape[:-3] + depth.shape[-2:]
        depth = depth.reshape(out_shape)

        return depth


    def save(self, file_name):
        pass
