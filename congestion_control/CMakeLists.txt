# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

add_library(
  rl_congestion_control STATIC
  CongestionControlEnv.cpp
  CongestionControlEnvConfig.cpp
  CongestionControlLocalEnv.cpp
  NetworkState.cpp
  RLCongestionController.cpp
)

if(INFERENCE_ONLY)
  target_compile_definitions(
    rl_congestion_control PUBLIC
    MVFSTRL_INFERENCE_ONLY
  )
else()
  # Add one more cpp
  target_sources(
    rl_congestion_control PRIVATE
    CongestionControlRPCEnv.cpp
  )
  add_dependencies(
    rl_congestion_control
    rpcenv_pb
  )
  target_link_libraries(
    rl_congestion_control PUBLIC
    rpcenv_pb
  )
endif()

target_compile_options(
  rl_congestion_control
  PRIVATE
  ${_QUIC_COMMON_COMPILE_OPTIONS}
)

add_dependencies(
  rl_congestion_control
  mvfst
)

target_include_directories(
  rl_congestion_control PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  rl_congestion_control PUBLIC
  mvfst
  ${TORCH_LIBRARIES}
)

target_compile_definitions(
  rl_congestion_control PUBLIC
  C10_USE_GLOG  # Fix glog compilation issues related to PyTorch
)
