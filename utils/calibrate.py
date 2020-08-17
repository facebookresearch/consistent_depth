#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from numpy.linalg import inv
import cv2
from sklearn import linear_model


def resize_small(gt, x, interp=cv2.INTER_NEAREST):
    """
    Resize to match the smaller image.
    """
    def size(x):
        return x.shape[:2][::-1]

    size_gt, size_x = size(gt), size(x)

    if size_gt == size_x:
        return gt, x

    if np.prod(size_gt) < np.prod(size_x):
        x = cv2.resize(x, size_gt, interpolation=interp)
    else:
        gt = cv2.resize(gt, size_x, interpolation=interp)
    return gt, x


# calibration
def calibrate_scale_shift(gt, x):
    ix = np.isfinite(gt) & np.isfinite(x)
    gt = gt[ix]
    x = x[ix]
    x2s = (x * x).flatten().sum()
    xs = x.flatten().sum()
    os = np.ones_like(x.flatten()).sum()
    xgs = (x * gt).flatten().sum()
    gs = gt.sum()
    A = np.array([
        [x2s, xs],
        [xs, os]
    ])
    b = np.array(
        [xgs, gs]
    ).T
    s, t = inv(A).dot(b)
    return np.array([s, t])


def calibrate_scale_shift_RANSAC(
    gt, x, max_trials=100000, stop_prob=0.999
):
    ix = np.isfinite(gt) & np.isfinite(x)
    gt = gt[ix].reshape(-1, 1)
    x = x[ix].reshape(-1, 1)

    ransac = linear_model.RANSACRegressor(
        max_trials=max_trials, stop_probability=stop_prob
    )
    ransac.fit(x, gt)
    s = ransac.estimator_.coef_[0, 0]
    t = ransac.estimator_.intercept_[0]
    return s, t


def calibrate_scale(gt, x, reduce=np.median):
    ix = np.isfinite(gt) & np.isfinite(x)
    ratios = gt[ix] / x[ix]
    return reduce(ratios)


# conversion
def cvt_by_scale_shift(depth, calib_data):
    s, t = calib_data
    return depth * s + t


CALIB_METHOD_MAP = {
    "scale": calibrate_scale,
    "scale-shift": calibrate_scale_shift,
    "ransac": calibrate_scale_shift_RANSAC,
}


def calibrate(gt, x, method: str):
    return CALIB_METHOD_MAP[method](gt, x)
