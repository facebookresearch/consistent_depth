#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import itertools
import json
import math
import os
from os.path import join as pjoin
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from typing import Dict

from utils.helpers import SuppressedStdout
from monodepth.depth_model_registry import get_depth_model

import optimizer
from loaders.video_dataset import VideoDataset, VideoFrameDataset
from loss.joint_loss import JointLoss
from loss.loss_params import LossParams
from utils import image_io, visualization
from utils.torch_helpers import to_device


class DepthFineTuningParams:
    """Options about finetune parameters.
    """

    @staticmethod
    def add_arguments(parser):
        parser = LossParams.add_arguments(parser)

        parser.add_argument(
            "--optimizer",
            default="Adam",
            choices=optimizer.OPTIMIZER_NAMES,
            help="optimizer to train the network",
        )
        parser.add_argument(
            "--val_epoch_freq",
            type=int,
            default=1,
            help="validation epoch frequency.",
        )
        parser.add_argument("--learning_rate", type=float, default=0,
            help="Learning rate for the training. If <= 0 it will be set"
            " automatically to the default for the specified model adapter.")
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_epochs", type=int, default=20)

        parser.add_argument("--log_dir", help="folder to log tensorboard summary")

        parser.add_argument('--display_freq', type=int, default=100,
            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1,
            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
            help='frequency of saving checkpoints at the end of epochs')

        return parser


def log_loss_stats(
    writer: SummaryWriter,
    name_prefix: str,
    loss_meta: Dict[str, torch.Tensor],
    n: int,
    log_histogram: bool = False,
):
    """
    loss_meta: sub_loss_name: individual losses
    """
    for sub_loss_name, loss_value in loss_meta.items():
        sub_loss_full_name = name_prefix + "/" + sub_loss_name

        writer.add_scalar(
            sub_loss_full_name + "/max", loss_value.max(), n,
        )
        writer.add_scalar(
            sub_loss_full_name + "/min", loss_value.min(), n,
        )
        writer.add_scalar(
            sub_loss_full_name + "/mean", loss_value.mean(), n,
        )

        if log_histogram:
            writer.add_histogram(sub_loss_full_name, loss_value, n)


def write_summary(
    writer, mode_name, input_images, depth, metadata, n_iter
):
    DIM = -3
    B = depth.shape[0]

    inv_depth_pred = depth.unsqueeze(DIM)

    mask = torch.stack(metadata['geometry_consistency']['masks'], dim=1)

    def to_vis(x):
        return x[:8].transpose(0, 1).reshape((-1,) + x.shape[DIM:])

    writer.add_image(
        mode_name + '/image',
        vutils.make_grid(to_vis(input_images), nrow=B, normalize=True), n_iter)
    writer.add_image(
        mode_name + '/pred_full',
        vutils.make_grid(to_vis(1.0 / inv_depth_pred), nrow=B, normalize=True), n_iter)
    writer.add_image(
        mode_name + '/mask',
        vutils.make_grid(to_vis(mask), nrow=B, normalize=True), n_iter)


def log_loss(
    writer: SummaryWriter,
    mode_name: str,
    loss: torch.Tensor,
    loss_meta: Dict[str, torch.Tensor],
    niters: int,
):
    main_loss_name = mode_name + "/loss"

    writer.add_scalar(main_loss_name, loss, niters)
    log_loss_stats(writer, main_loss_name, loss_meta, niters)


def make_tag(params):
    return (
        LossParams.make_str(params)
        + f"_LR{params.learning_rate}"
        + f"_BS{params.batch_size}"
        + f"_O{params.optimizer.lower()}"
    )


class DepthFineTuner:
    def __init__(self, range_dir, frames, params):
        self.frames = frames
        self.params = params
        self.base_dir = params.path
        self.range_dir = range_dir
        self.out_dir = pjoin(self.range_dir, make_tag(params))
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"Fine-tuning directory: '{self.out_dir}'")

        self.checkpoints_dir = pjoin(self.out_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        model = get_depth_model(params.model_type)
        self.model = model()

        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs.")
        if num_gpus > 1:
            self.params.batch_size *= num_gpus
            print(f"Adjusting batch size to {self.params.batch_size}.")

        self.reference_disparity = {}
        self.vis_depth_scale = None

    def save_depth(self, dir: str = None, frames=None):
        if dir is None:
            dir = self.out_dir
        if frames is None:
            frames = self.frames

        color_fmt = pjoin(self.base_dir, "color_down", "frame_{:06d}.raw")
        depth_dir = pjoin(dir, "depth")
        depth_fmt = pjoin(depth_dir, "frame_{:06d}")

        dataset = VideoFrameDataset(color_fmt, frames)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model.eval()

        os.makedirs(depth_dir, exist_ok=True)
        for data in data_loader:
            data = to_device(data)
            stacked_images, metadata = data
            frame_id = metadata["frame_id"][0]

            depth = self.model.forward(stacked_images, metadata)

            depth = depth.detach().cpu().numpy().squeeze()
            inv_depth = 1.0 / depth

            image_io.save_raw_float32_image(
                depth_fmt.format(frame_id) + ".raw", inv_depth)

        with SuppressedStdout():
            visualization.visualize_depth_dir(depth_dir, depth_dir, force=True)

    def fine_tune(self, writer=None):
        meta_file = pjoin(self.range_dir, "metadata_scaled.npz")

        dataset = VideoDataset(self.base_dir, meta_file)
        train_data_loader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        val_data_loader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        criterion = JointLoss(self.params,
            parameters_init=[p.clone() for p in self.model.parameters()])

        if writer is None:
            log_dir = pjoin(self.out_dir, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

        opt = optimizer.create(
            self.params.optimizer,
            self.model.parameters(),
            self.params.learning_rate,
            betas=(0.9, 0.999)
        )

        eval_dir = pjoin(self.out_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        self.model.train()

        def suffix(epoch, niters):
            return "_e{:04d}_iter{:06d}".format(epoch, niters)

        def validate(epoch, niters):
            loss_meta = self.eval_and_save(
                criterion, val_data_loader, suffix(epoch, niters)
            )
            if writer is not None:
                log_loss_stats(
                    writer, "validation", loss_meta, epoch, log_histogram=True
                )
            print(f"Done Validation for epoch {epoch} ({niters} iterations)")

        self.vis_depth_scale = None
        validate(0, 0)

        # Training loop.
        total_iters = 0
        for epoch in range(self.params.num_epochs):
            epoch_start_time = time.perf_counter()

            for data in train_data_loader:
                data = to_device(data)
                stacked_img, metadata = data

                depth = self.model(stacked_img, metadata)

                opt.zero_grad()
                loss, loss_meta = criterion(
                    depth, metadata, parameters=self.model.parameters())

                pairs = metadata['geometry_consistency']['indices']
                pairs = pairs.cpu().numpy().tolist()

                print(f"Epoch = {epoch}, pairs = {pairs}, loss = {loss[0]}")
                if torch.isnan(loss):
                    print("Loss is NaN. Skipping.")
                    continue

                loss.backward()
                opt.step()

                total_iters += stacked_img.shape[0]

                if writer is not None and total_iters % self.params.print_freq == 0:
                    log_loss(writer, 'Train', loss, loss_meta, total_iters)

                if writer is not None and total_iters % self.params.display_freq == 0:
                    write_summary(
                        writer, 'Train', stacked_img, depth, metadata, total_iters
                    )

            epoch_end_time = time.perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch} took {epoch_duration:.2f}s.")

            if (epoch + 1) % self.params.val_epoch_freq == 0:
                validate(epoch + 1, total_iters)

            if (epoch + 1) % self.params.save_epoch_freq == 0:
                file_name = pjoin(self.checkpoints_dir, f"{epoch + 1:04d}.pth")
                self.model.save(file_name)

        # Validate the last epoch, unless it was just done in the loop above.
        if self.params.num_epochs % self.params.val_epoch_freq != 0:
            validate(self.params.num_epochs, total_iters)

        print("Finished Training")

    def eval_and_save(self, criterion, data_loader, suf) -> Dict[str, torch.Tensor]:
        """
        Note this function asssumes the structure of the data produced by data_loader
        """
        N = len(data_loader.dataset)
        loss_dict = {}
        saved_frames = set()
        total_index = 0
        max_frame_index = 0
        all_pairs = []

        for _, data in zip(range(N), data_loader):
            data = to_device(data)
            stacked_img, metadata = data

            with torch.no_grad():
                depth = self.model(stacked_img, metadata)

            batch_indices = (
                metadata["geometry_consistency"]["indices"].cpu().numpy().tolist()
            )

            # Update the maximum frame index and pairs list.
            max_frame_index = max(max_frame_index, max(itertools.chain(*batch_indices)))
            all_pairs += batch_indices

            # Compute and store losses.
            _, loss_meta = criterion(
                depth, metadata, parameters=self.model.parameters(),
            )

            for loss_name, losses in loss_meta.items():
                if loss_name not in loss_dict:
                    loss_dict[loss_name] = {}

                for indices, loss in zip(batch_indices, losses):
                    loss_dict[loss_name][str(indices)] = loss.item()

            # Save depth maps.
            inv_depths_batch = 1.0 / depth.cpu().detach().numpy()
            if self.vis_depth_scale is None:
                # Single scale for the whole dataset.
                self.vis_depth_scale = inv_depths_batch.max()

            for inv_depths, indices in zip(inv_depths_batch, batch_indices):
                for inv_depth, index in zip(inv_depths, indices):
                    # Only save frames not saved before.
                    if index in saved_frames:
                        continue
                    saved_frames.add(index)

                    fn_pre = pjoin(
                        self.out_dir, "eval", "depth_{:06d}{}".format(index, suf)
                    )
                    image_io.save_raw_float32_image(fn_pre + ".raw", inv_depth)

                    inv_depth_vis = visualization.visualize_depth(
                        inv_depth, depth_min=0, depth_max=self.vis_depth_scale
                    )
                    cv2.imwrite(fn_pre + ".png", inv_depth_vis)
                total_index += 1

        loss_meta = {
            loss_name: torch.tensor(tuple(loss.values()))
            for loss_name, loss in loss_dict.items()
        }
        loss_dict["mean"] = {k: v.mean().item() for k, v in loss_meta.items()}

        with open(pjoin(self.out_dir, "eval", "loss{}.json".format(suf)), "w") as f:
            json.dump(loss_dict, f)

        # Print verbose summary to stdout.
        index_width = int(math.ceil(math.log10(max_frame_index)))
        loss_names = list(loss_dict.keys())
        loss_names.remove("mean")
        loss_format = {}
        for name in loss_names:
            max_value = max(loss_dict[name].values())
            width = math.ceil(math.log10(max_value))
            loss_format[name] = f"{width+7}.6f"

        for pair in sorted(all_pairs):
            line = f"({pair[0]:{index_width}d}, {pair[1]:{index_width}d}): "
            line += ", ".join(
                [f"{name}: {loss_dict[name][str(pair)]:{loss_format[name]}}"
                for name in loss_names]
            )
            print(line)

        print("Mean: " + " " * (2 * index_width) + ", ".join(
            [f"{name}: {loss_dict[name][str(pair)]:{loss_format[name]}}"
            for name in loss_names]
        ))

        return loss_meta
