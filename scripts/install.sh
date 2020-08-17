# Copyright (c) Facebook, Inc. and its affiliates.

# conda create -n consistent_depth python=3.6
# conda activate consistent_depth
# Note to set LD_LIBRARY_PATH if conda activate fails to do so.
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pushd third_party/flownet2
chmod +x install.sh
./install.sh
popd
