#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import numpy as np
import os
from PIL import Image
import cv2
import struct
from subprocess import call
import warnings
import six

if six.PY2:

    class ResourceWarning(RuntimeWarning):
        pass


# Needed to suppress ResourceWarning for unclosed image file on dev server.
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", UserWarning)


# resizes the image
def resize_to_target(image, max_size, align=1, suppress_messages=False):
    if not suppress_messages:
        print("Original size: %d x %d" % (image.shape[1], image.shape[0]))

    H, W = image.shape[:2]
    long_side = float(max(W, H))
    scale = min(1.0, max_size / long_side)
    resized_height = int(H * scale)
    resized_width = int(W * scale)
    if resized_width % align != 0:
        resized_width = align * round(resized_width / align)
        if not suppress_messages:
            print("Rounding width to closest multiple of %d." % align)
    if resized_height % align != 0:
        resized_height = align * round(resized_height / align)
        if not suppress_messages:
            print("Rounding height to closest multiple of %d." % align)

    if not suppress_messages:
        print("Resized: %d x %d" % (resized_width, resized_height))
    image = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    return image


# Reads an image and returns a normalized float buffer (0-1 range). Corrects
# rotation based on EXIF tags.
def load_image(file_name, max_size=None, align=1, suppress_messages=False):
    img, angle = load_image_angle(
        file_name, max_size, align=align, suppress_messages=suppress_messages
    )
    return img


def load_image_angle(
    file_name, max_size=None, min_size=None,
    angle=0, align=1, suppress_messages=False
):
    with Image.open(file_name) as img:
        if hasattr(img, "_getexif") and img._getexif() is not None:
            # orientation tag in EXIF data is 274
            exif = dict(img._getexif().items())

            # adjust the rotation
            if 274 in exif:
                if exif[274] == 8:
                    angle = 90
                elif exif[274] == 6:
                    angle = 270
                elif exif[274] == 3:
                    angle = 180

        if angle != 0:
            img = img.rotate(angle, expand=True)

        img = np.float32(img) / 255.0

        if max_size is not None:
            if min_size is not None:
                img = cv2.resize(
                    img, (max_size, min_size), interpolation=cv2.INTER_AREA)
            else:
                img = resize_to_target(
                    img, max_size, align=align, suppress_messages=suppress_messages
                )

        return img, angle

    return [[]], 0.0


# Load image from binary file in the same way as read in C++ with
# #include "compphotolib/core/CvUtil.h"
# freadimg(fileName, image);
def load_raw_float32_image(file_name):
    with open(file_name, "rb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5
        I_BYTES = 4
        Q_BYTES = 8

        h = struct.unpack("i", f.read(I_BYTES))[0]
        w = struct.unpack("i", f.read(I_BYTES))[0]

        cv_type = struct.unpack("i", f.read(I_BYTES))[0]
        pixel_size = struct.unpack("Q", f.read(Q_BYTES))[0]
        d = ((cv_type - CV_32F) >> CV_CN_SHIFT) + 1
        assert d >= 1
        d_from_pixel_size = pixel_size // 4
        if d != d_from_pixel_size:
            raise Exception(
                "Incompatible pixel_size(%d) and cv_type(%d)" % (pixel_size, cv_type)
            )
        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")

        data = np.frombuffer(f.read(), dtype=np.float32)
        result = data.reshape(h, w) if d == 1 else data.reshape(h, w, d)
        return result


# Save image to binary file, so that it can be read in C++ with
# #include "compphotolib/core/CvUtil.h"
# freadimg(fileName, image);
def save_raw_float32_image(file_name, image):
    with open(file_name, "wb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5

        dims = image.shape
        h = 0
        w = 0
        d = 1
        if len(dims) == 2:
            h, w = image.shape
            float32_image = np.transpose(image).astype(np.float32)
        else:
            h, w, d = image.shape
            float32_image = np.transpose(image, [2, 1, 0]).astype("float32")

        cv_type = CV_32F + ((d - 1) << CV_CN_SHIFT)

        pixel_size = d * 4

        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")
        f.write(struct.pack("i", h))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", cv_type))
        f.write(struct.pack("Q", pixel_size))  # Write size_t ~ uint64_t

        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024 ** 2 // image.itemsize, 1)

        for chunk in np.nditer(
            float32_image,
            flags=["external_loop", "buffered", "zerosize_ok"],
            buffersize=buffersize,
            order="F",
        ):
            f.write(chunk.tobytes("C"))


def save_image(file_name, image):
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".raw":
        save_raw_float32_image(file_name, image)
    else:
        image = 255.0 * image
        image = Image.fromarray(image.astype("uint8"))
        image.save(file_name)


def save_depth_map_colored(file_name, depth_map, color_binary):
    save_image(file_name, depth_map)
    color_depth_name = os.path.splitext(file_name)[0] + "_color.jpg"
    if color_binary != "":
        call([color_binary, "--inputFile", file_name, "--outputFile", color_depth_name])


# main print_function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="input image")
    parser.add_argument("--output_image", type=str, help="output image")
    parser.add_argument(
        "--max_size", type=int, default=768, help="max size of long image dimension"
    )
    args, unknown = parser.parse_known_args()

    img = load_image(args.input_image, int(args.max_size))
    save_image(args.output_image, img)
