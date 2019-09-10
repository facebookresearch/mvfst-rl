#!/bin/bash

ROOT=$(pwd)

PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

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
