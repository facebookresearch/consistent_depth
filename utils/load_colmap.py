#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from third_party.colmap.scripts.python.read_write_model import (
    CAMERA_MODELS,
    rotmat2qvec,
    Camera,
    BaseImage,
    write_model
)
# for exporting these functions to the rest of the code
from third_party.colmap.scripts.python.read_dense import read_array
from third_party.colmap.scripts.python.read_write_model import (
    qvec2rotmat,
    read_images_binary,
    read_points3d_binary,
    read_cameras_binary,
    read_model,
)


CAMERA_NAME_TO_IDS = {
    c.model_name: c.model_id for c in CAMERA_MODELS
}

# maps colmap point xc to normal coordinate frame x
# x = ROT_COLMAP_TO_NORMAL * x
ROT_COLMAP_TO_NORMAL = np.diag([1, -1, -1])


def intrinsics_to_camera(intrinsics, src_im_size=None, dst_im_size=None, eps=0.01):
    """Convert metadata intrinsics to COLMAP Camera.
    Only support shared SIMPLE_PINHOLE camera.
    Args:
        intrinsics: (N, 4) where each row is fx, fy, cx, cy.
            Assume intrinsics is the same across all frames.
        src_im_size: image size corresponding to intrinsics
        dst_im_size: the image size we want to convert to
    """
    fxy, cxy = intrinsics[0][:2], intrinsics[0][-2:]

    if src_im_size is None:
        src_im_size = (2 * cxy).astype(int)

    if dst_im_size is None:
        dst_im_size = src_im_size

    ratio = np.array(dst_im_size) / np.array(src_im_size).astype(float)
    fxy *= ratio
    cxy *= ratio

    if np.abs(fxy[0] - fxy[1]) < eps:
        model = 'SIMPLE_PINHOLE'
        params = np.array((fxy[0], cxy[0], cxy[1]))
    else:
        model = 'PINHOLE'
        params = np.array((fxy[0], fxy[1], cxy[0], cxy[1]))

    camera = Camera(
        id=1, model=model,
        width=dst_im_size[0], height=dst_im_size[1],
        params=params
    )
    return {camera.id: camera}


def extrinsics_to_images(extrinsics):
    images = {}
    for i, extr in enumerate(extrinsics):
        R, t = extr[:, :3], extr[:, -1:]
        Rc = ROT_COLMAP_TO_NORMAL.dot(R.T).dot(ROT_COLMAP_TO_NORMAL.T)
        tc = -Rc.dot(ROT_COLMAP_TO_NORMAL.T).dot(t)

        frame_id = i + 1
        image = BaseImage(
            id=frame_id, qvec=rotmat2qvec(Rc), tvec=tc.flatten(),
            camera_id=1, name="frame_%06d.png" % i,
            xys=[], point3D_ids=[]
        )
        images[image.id] = image
    return images


def to_colmap(intrinsics, extrinsics, src_im_size=None, dst_im_size=None):
    """Convert Extrinsics and intrinsics to an empty COLMAP project with no points.
    """
    cameras = intrinsics_to_camera(
        intrinsics, src_im_size=src_im_size, dst_im_size=dst_im_size
    )
    images = extrinsics_to_images(extrinsics)
    points3D = {}
    return cameras, images, points3D


def save_colmap(
    path, intrinsics, extrinsics, src_im_size=None, dst_im_size=None, ext=".txt"
):
    cameras, images, points3D = to_colmap(intrinsics, extrinsics,
        src_im_size=src_im_size, dst_im_size=dst_im_size)
    write_model(cameras, images, points3D, path, ext)


def cameras_to_intrinsics(cameras, camera_ids, size_new):
    """
    Args:
        size_new: image size after resizing and produce equivalent intrinsics
        for this size
    """
    # params = f, cx, cy
    assert all(
        (c.model == "SIMPLE_PINHOLE" or c.model == "PINHOLE"
            or c.model == "SIMPLE_RADIAL"
         for c in cameras.values()))

    intrinsics = []
    for id in camera_ids:
        c = cameras[id]
        if c.model == "SIMPLE_PINHOLE":
            f, cx, cy = c.params
            fxy = np.array([f, f])
        elif c.model == "PINHOLE":
            fx, fy, cx, cy = c.params
            fxy = np.array([fx, fy])
        elif c.model == "SIMPLE_RADIAL":
            f, cx, cy, r = c.params
            fxy = np.array([f, f])
        else:
            raise AssertionError()
        ratio = np.array(size_new) / np.array((c.width, c.height))
        fxy = fxy * ratio
        cxy = np.array((cx, cy)) * ratio
        intrinsics.append(np.concatenate((fxy, cxy)))
    return np.stack(intrinsics, axis=0)


def images_to_extrinsics(images, image_ids):
    """Let p be in local camera coordinates. x in global coordinates.
    Rc, tc be rotation and translation from colmap
    p = Rc * x + tc, i.e., x = Rc^T * p - Rc^T * tc
    But we want to generate R, t, s.t.,
    x = Rx+t,
    so R = Rc^T, t = - Rc^T * tc

    Note that colmap uses a different coordinate system where y points down and
    z points to the world.
    """
    extrinsics = []
    for id in image_ids:
        im = images[id]
        Rc, tc = im.qvec2rotmat(), im.tvec
        R, t = Rc.T, -Rc.T.dot(tc.reshape(-1, 1))
        R = ROT_COLMAP_TO_NORMAL.dot(R).dot(ROT_COLMAP_TO_NORMAL.T)
        t = ROT_COLMAP_TO_NORMAL.dot(t)
        extrinsics.append(np.concatenate([R, t], axis=1))
    return np.stack(extrinsics, axis=0)


def convert_points3D(pts3D: np.ndarray):
    """
    points (3, N)
    """
    return ROT_COLMAP_TO_NORMAL.dot(pts3D)


def ordered_image_ids(images):
    return sorted(images.keys(), key=lambda id: images[id].name)


def convert_calibration(cameras, images, size_new):
    sorted_im_ids = ordered_image_ids(images)
    sorted_cam_ids = [images[id].camera_id for id in sorted_im_ids]
    intrinsics = cameras_to_intrinsics(cameras, sorted_cam_ids, size_new)
    extrinsics = images_to_extrinsics(images, sorted_im_ids)
    return intrinsics, extrinsics
