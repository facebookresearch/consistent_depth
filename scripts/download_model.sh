# Copyright (c) Facebook, Inc. and its affiliates.

#!/bin/bash

set -exo
mkdir -p checkpoints
gdown https://drive.google.com/uc?id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da -O checkpoints/flownet2.pth
