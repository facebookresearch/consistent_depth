# Copyright (c) Facebook, Inc. and its affiliates.

set -exo

results_dir="$1"

mkdir -p data/videos/
wget https://www.dropbox.com/s/9a2kb7flg3o1eb5/ayush_color.mp4?dl=1 -O data/videos/ayush.mp4

mkdir -p "${results_dir}"
wget https://www.dropbox.com/s/7mbvu60qbs7hzod/ayush_colmap.zip?dl=1 -O "${results_dir}/ayush_colmap.zip"
unzip "${results_dir}/ayush_colmap.zip" -d "${results_dir}"
rm "${results_dir}/ayush_colmap.zip"
