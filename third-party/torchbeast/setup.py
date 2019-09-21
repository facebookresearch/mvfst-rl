# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# CXX=c++ python3 setup.py build develop
#   or
# CXX=c++ pip install . -vv
#
# Potentially also set TORCHBEAST_LIBS_PREFIX.

import sys
import subprocess
import setuptools
import os

from torch.utils import cpp_extension


PREFIX = os.getenv("CONDA_PREFIX")

if os.getenv("TORCHBEAST_LIBS_PREFIX"):
    PREFIX = os.getenv("TORCHBEAST_LIBS_PREFIX")
if not PREFIX:
    PREFIX = "/usr/local"


extra_compile_args = []
extra_link_args = []

protoc = f"{PREFIX}/bin/protoc"

grpc_objects = [
    f"{PREFIX}/lib/libgrpc++.a",
    f"{PREFIX}/lib/libgrpc.a",
    f"{PREFIX}/lib/libgpr.a",
    f"{PREFIX}/lib/libaddress_sorting.a",
]

include_dirs = cpp_extension.include_paths() + [f"{PREFIX}/include"]
libraries = []

if sys.platform == "darwin":
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
    extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]

    # Relevant only when c-cares is not embedded in grpc, e.g. when
    # installing grpc via homebrew.
    libraries.append("cares")
elif sys.platform == "linux":
    libraries.append("z")

grpc_objects.append(f"{PREFIX}/lib/libprotobuf.a")


actorpool = cpp_extension.CppExtension(
    name="libtorchbeast.actorpool",
    sources=[
        "libtorchbeast/actorpool.cc",
        "libtorchbeast/rpcenv.pb.cc",
        "libtorchbeast/rpcenv.grpc.pb.cc",
    ],
    include_dirs=include_dirs,
    libraries=libraries,
    language="c++",
    extra_compile_args=["-std=c++17"] + extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=grpc_objects,
)


def build_pb():
    print("calling protoc")
    if (
        subprocess.call(
            [protoc, "--cpp_out=libtorchbeast", "-Ilibtorchbeast", "rpcenv.proto"]
        )
        != 0
    ):
        sys.exit(-1)
    if (
        subprocess.call(
            protoc + " --grpc_out=libtorchbeast -Ilibtorchbeast"
            " --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`"
            " rpcenv.proto",
            shell=True,
        )
        != 0
    ):
        sys.exit(-1)


build_pb()
setuptools.setup(
    name="libtorchbeast",
    packages=["libtorchbeast"],
    version="0.0.8",
    ext_modules=[actorpool],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
