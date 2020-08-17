#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import copy
import cv2
import numpy as np
import os
import torch

from third_party.flownet2.models import FlowNet2
from third_party.OpticalFlowToolkit.lib.flowlib import flow_to_image
from utils.image_io import save_raw_float32_image


class FlowInfer(torch.utils.data.Dataset):
    def __init__(self, list_file, size=None, isRGB=True, start_pos=0):
        super(FlowInfer, self).__init__()
        self.size = size
        txt_file = open(list_file, "r")
        self.frame1_list = []
        self.frame2_list = []
        self.output_list = []
        self.isRGB = isRGB

        for line in txt_file:
            line = line.strip(" ")
            line = line.strip("\n")

            line_split = line.split(" ")
            self.frame1_list.append(line_split[0])
            self.frame2_list.append(line_split[1])
            self.output_list.append(line_split[2])

        if start_pos > 0:
            self.frame1_list = self.frame1_list[start_pos:]
            self.frame2_list = self.frame2_list[start_pos:]
            self.output_list = self.output_list[start_pos:]
        txt_file.close()

    def __len__(self):
        return len(self.frame1_list)

    def __getitem__(self, idx):
        frame1 = cv2.imread(self.frame1_list[idx])
        frame2 = cv2.imread(self.frame2_list[idx])
        if self.isRGB:
            frame1 = frame1[:, :, ::-1]
            frame2 = frame2[:, :, ::-1]
        output_path = self.output_list[idx]

        frame1 = self._img_tf(frame1)
        frame2 = self._img_tf(frame2)

        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).contiguous().float()
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).contiguous().float()

        return frame1_tensor, frame2_tensor, output_path

    def _img_tf(self, img):
        img = cv2.resize(img, (self.size[1], self.size[0]))

        return img


def detectAndDescribe(image):
    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SURF_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None


def parse_args():
    parser = argparse.ArgumentParser("Compute optical flow from im1 to im2")

    parser.add_argument("--im1", nargs="+")
    parser.add_argument("--im2", nargs="+")
    parser.add_argument("--out", nargs="+")
    parser.add_argument(
        "--pretrained_model_flownet2",
        type=str,
        default="./pretrained_models/FlowNet2_checkpoint.pth.tar",
    )
    # parser.add_argument('--img_size', type=list, default=(512, 1024, 3))
    parser.add_argument("--rgb_max", type=float, default=255.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--homography", type=bool, default=1)
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=None,
        help="If size is not None, resize the flow to size."
        + " O.w., resize based on max_size and divide.",
    )
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--divide", type=int, default=1)
    parser.add_argument("--visualize", type=bool, default=False)

    args = parser.parse_args()
    return args


def getimage(img1_path, img2_path, size=None):
    frame1 = cv2.imread(img1_path)
    frame2 = cv2.imread(img2_path)

    if size is not None:
        frame1 = cv2.resize(frame1[:, :, ::-1], (size[1], size[0]))
        frame2 = cv2.resize(frame2[:, :, ::-1], (size[1], size[0]))

    imgH, imgW, _ = frame1.shape

    (kpsA, featuresA) = detectAndDescribe(frame1)
    (kpsB, featuresB) = detectAndDescribe(frame2)
    try:
        (_, H_BA, _) = matchKeypoints(kpsB, kpsA, featuresB, featuresA)
    except Exception:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)

    NoneType = type(None)
    if type(H_BA) == NoneType:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)

    try:
        np.linalg.inv(H_BA)
    except Exception:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)

    img2_registered = cv2.warpPerspective(frame2, H_BA, (imgW, imgH))

    frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).contiguous().float()
    frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).contiguous().float()
    frame2_reg_tensor = (
        torch.from_numpy(img2_registered).permute(2, 0, 1).contiguous().float()
    )

    return frame1_tensor, frame2_tensor, frame2_reg_tensor, H_BA


def infer(args, Flownet, device, img1_name, img2_name):
    img1, img2, img2_reg, H_BA = getimage(img1_name, img2_name)
    _, imgH, imgW = img1.shape
    img1 = img1[None, :, :]
    img2 = img2[None, :, :]
    img2_reg = img2_reg[None, :, :]
    img1 = img1.to(device)
    img2 = img2.to(device)
    img2_reg = img2_reg.to(device)

    if args.homography != 1:
        sz = img1.size()
        img1_view = img1.view(sz[0], sz[1], 1, sz[2], sz[3])
        img2_view = img2.view(sz[0], sz[1], 1, sz[2], sz[3])
        inputs = torch.cat((img1_view, img2_view), dim=2)
        flow = Flownet(inputs)[0].permute(1, 2, 0).data.cpu().numpy()
    else:
        sz = img1.size()
        img1_view = img1.view(sz[0], sz[1], 1, sz[2], sz[3])
        img2_reg_view = img2_reg.view(sz[0], sz[1], 1, sz[2], sz[3])
        inputs = torch.cat((img1_view, img2_reg_view), dim=2)
        flow = Flownet(inputs)[0].permute(1, 2, 0).data.cpu().numpy()

        (fy, fx) = np.mgrid[0:imgH, 0:imgW].astype(np.float32)

        fxx = copy.deepcopy(fx) + flow[:, :, 0]
        fyy = copy.deepcopy(fy) + flow[:, :, 1]

        (fxxx, fyyy, fz) = np.linalg.inv(H_BA).dot(
            np.concatenate(
                (
                    fxx.reshape(1, -1),
                    fyy.reshape(1, -1),
                    np.ones_like(fyy).reshape(1, -1),
                ),
                axis=0,
            )
        )
        fxxx, fyyy = fxxx / fz, fyyy / fz

        flow = np.concatenate(
            (
                fxxx.reshape(imgH, imgW, 1) - fx.reshape(imgH, imgW, 1),
                fyyy.reshape(imgH, imgW, 1) - fy.reshape(imgH, imgW, 1),
            ),
            axis=2,
        )

    return flow


def resize_flow(flow, size):
    resized_width, resized_height = size
    H, W = flow.shape[:2]
    scale = np.array((resized_width / float(W), resized_height / float(H))).reshape(
        1, 1, -1
    )
    resized = cv2.resize(
        flow, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC
    )
    resized *= scale
    return resized


def process(args):
    N = len(args.im1)
    assert N == len(args.im2) and N == len(args.out)

    device = torch.device("cuda:0")
    Flownet = FlowNet2(args)
    print(f"Loading pretrained model from '{args.pretrained_model_flownet2}'.")
    flownet2_ckpt = torch.load(args.pretrained_model_flownet2)
    Flownet.load_state_dict(flownet2_ckpt["state_dict"])
    Flownet.to(device)
    Flownet.eval()

    for im1, im2, out in zip(args.im1, args.im2, args.out):
        if os.path.isfile(out):
            continue

        flow = infer(args, Flownet, device, im1, im2)
        flow = resize_flow(flow, args.size)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_raw_float32_image(out, flow)

        if args.visualize:
            vis = flow_to_image(flow)
            cv2.imwrite(os.path.splitext(out)[0] + ".png", vis)


if __name__ == "__main__":
    args = parse_args()
    process(args)
