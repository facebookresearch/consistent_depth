# Copyright (c) Facebook, Inc. and its affiliates.

# Install packages
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

mkdir -p colmap-packages
pushd colmap-packages

# Install ceres-solver [10-20 min]
sudo apt-get install libatlas-base-dev libsuitesparse-dev
git clone https://ceres-solver.googlesource.com/ceres-solver
pushd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make
sudo make install
popd  # pop ceres-solver

# Install colmap-3.6
git clone https://github.com/colmap/colmap
pushd colmap
git checkout dev
git checkout tags/3.6-dev.3 -b dev-3.6
mkdir build
cd build
cmake ..
make
sudo make install
CC=/usr/bin/gcc-6 CXX=/usr/bin/g++-6 cmake ..
popd  # pop colmap

popd
