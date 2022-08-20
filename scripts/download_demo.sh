# Copyright (c) Facebook, Inc. and its affiliates.

set -exo

results_dir="$1"

mkdir -p data/videos/
gdown https://drive.google.com/uc?id=1y6_L3uXwsDQV-_ajmba1hWuS8GFZ75_8 -O data/videos/ayush.mp4

mkdir -p "${results_dir}"
gdown https://drive.google.com/uc?id=1jamL_Gv4DjodXiVx1U9ze__Nv9tER3LH -O "${results_dir}/ayush_colmap.zip"
unzip "${results_dir}/ayush_colmap.zip" -d "${results_dir}"
rm "${results_dir}/ayush_colmap.zip"
