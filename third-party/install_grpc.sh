#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

ROOT=$(pwd)

PREFIX=${CONDA_PREFIX:-"/usr/local"}

NPROCS=$(getconf _NPROCESSORS_ONLN)

cd ${ROOT}/third-party/grpc

## This requires libprotobuf to be installed in the conda env.
## Otherwise, we could also do this:
# cd ${ROOT}/third-party/grpc/third_party/protobuf
# ./autogen.sh && ./configure --prefix=${PREFIX}
# make && make install && ldconfig

# Make make find libprotobuf
export CPATH=${PREFIX}/include:${CPATH}
export LIBRARY_PATH=${PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${PREFIX}/lib:${LD_LIBRARY_PATH}

make -j ${NPROCS} prefix=${PREFIX} \
     HAS_SYSTEM_PROTOBUF=true HAS_SYSTEM_CARES=false
make prefix=${PREFIX} \
     HAS_SYSTEM_PROTOBUF=true HAS_SYSTEM_CARES=false install

cd ${ROOT}
