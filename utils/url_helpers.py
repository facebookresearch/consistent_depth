#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
from os.path import join as pjoin
import wget
from zipfile import ZipFile


def get_model_from_url(
    url: str, local_path: str, is_zip: bool = False, path_root: str = "checkpoints"
) -> str:
    local_path = pjoin(path_root, local_path)
    if os.path.exists(local_path):
        print(f"Found cache {local_path}")
        return local_path

    # download
    local_path = local_path.rstrip(os.sep)
    download_path = local_path if not is_zip else f"{local_path}.zip"
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if os.path.isfile(download_path):
        print(f"Found cache {download_path}")
    else:
        print(f"Dowloading {url} to {download_path} ...")
        wget.download(url, download_path)

    if is_zip:
        print(f"Unziping {download_path} to {local_path}")
        with ZipFile(download_path, 'r') as f:
            f.extractall(local_path)
        os.remove(download_path)

    return local_path
