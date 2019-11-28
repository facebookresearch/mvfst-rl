# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
cmake_minimum_required(VERSION 3.10)

project(
  mvfst-rl
)

find_package(mvfst)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(QUIC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third-party/mvfst)
set(TORCHBEAST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third-party/torchbeast)

message(STATUS "PROJECT_ROOT: ${PROJECT_ROOT}")
message(STATUS "QUIC_ROOT: ${QUIC_ROOT}")
message(STATUS "TORCHBEAST_ROOT: ${TORCHBEAST_ROOT}")
message(STATUS "PREFIX_PATH: ${PREFIX_PATH}")
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "INFERENCE_ONLY: ${INFERENCE_ONLY}")

list(APPEND
  CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${QUIC_ROOT}/cmake
)

set(CMAKE_CXX_STANDARD 14)

list(APPEND
  _QUIC_BASE_COMPILE_OPTIONS
  -std=c++14
  -Wall
  -Wextra
)

list(APPEND
  _QUIC_COMMON_COMPILE_OPTIONS
  ${_QUIC_BASE_COMPILE_OPTIONS}
  # more strict options
  -Werror=sign-compare
  -Werror=bool-compare
  -Woverloaded-virtual
  -Wnon-virtual-dtor
  # disable noisy stuff
  -Wno-unused-parameter
  -Wno-narrowing
)

find_package(fmt REQUIRED)

# Find GFlags
SET(GFLAG_DEPENDENCIES "")
find_package(gflags CONFIG QUIET)
if (gflags_FOUND)
  message(STATUS "Found gflags from package config")
  if (TARGET gflags-shared)
    list(APPEND GFLAG_DEPENDENCIES gflags-shared)
  elseif (TARGET gflags)
    list(APPEND GFLAG_DEPENDENCIES gflags)
  else()
    message(FATAL_ERROR "Unable to determine the target name for the GFlags package.")
  endif()
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${GFLAGS_LIBRARIES})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${GFLAGS_INCLUDE_DIR})
else()
  find_package(Gflags REQUIRED MODULE)
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${LIBGFLAGS_LIBRARY})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${LIBGFLAGS_INCLUDE_DIR})
endif()

# Find GMock and GTest. Required for linking some TestUtils.
set(REQUIRED_LINK_DIRS "")
if(BUILD_TESTS)
  enable_testing()
  list(APPEND CMAKE_REQUIRED_INCLUDES "${QUIC_ROOT}/_build/build/googletest/src/googletest/googlemock/include")
  list(APPEND CMAKE_REQUIRED_INCLUDES "${QUIC_ROOT}/_build/build/googletest/src/googletest/googletest/include")
  list(APPEND REQUIRED_LINK_DIRS "${QUIC_ROOT}/_build/build/googletest/src/googletest-build/googlemock")
  list(APPEND REQUIRED_LINK_DIRS "${QUIC_ROOT}/_build/build/googletest/src/googletest-build/googlemock/gtest")
  list(APPEND CMAKE_REQUIRED_LIBRARIES "gmock")
  list(APPEND CMAKE_REQUIRED_LIBRARIES "gtest")
endif()

# Find PyTorch
find_package(Torch REQUIRED)
message(STATUS "Found PyTorch libs: ${TORCH_LIBRARIES}")

include_directories(
  ${PROJECT_ROOT}
  ${CMAKE_REQUIRED_INCLUDES}
  ${PREFIX_PATH}/include
  ${TORCHBEAST_ROOT}/libtorchbeast
)

link_directories(
  ${REQUIRED_LINK_DIRS}
  ${PREFIX_PATH}/lib
)

link_libraries(
  ${CMAKE_REQUIRED_LIBRARIES}
)

add_subdirectory(traffic_gen)
add_subdirectory(congestion_control)
add_subdirectory(third-party)
