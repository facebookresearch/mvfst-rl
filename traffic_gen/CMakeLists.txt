# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
add_executable(traffic_gen main.cpp)

target_compile_options(
  traffic_gen
  PRIVATE
  ${_QUIC_COMMON_COMPILE_OPTIONS}
)

add_dependencies(
  traffic_gen
  mvfst
  rl_congestion_control
)

target_include_directories(
  traffic_gen PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  traffic_gen PUBLIC
  mvfst
  rl_congestion_control
)
