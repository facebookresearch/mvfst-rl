add_library(
  rl_congestion_control STATIC
  CongestionControlEnv.cpp
  CongestionControlEnvConfig.cpp
  CongestionControlLocalEnv.cpp
  CongestionControlRPCEnv.cpp
  NetworkState.cpp
  RLCongestionController.cpp
)

target_compile_options(
  rl_congestion_control
  PRIVATE
  ${_QUIC_COMMON_COMPILE_OPTIONS}
)

add_dependencies(
  rl_congestion_control
  mvfst
  rpcenv_pb
)

target_include_directories(
  rl_congestion_control PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  rl_congestion_control PUBLIC
  mvfst
  ${TORCH_LIBRARIES}
  rpcenv_pb
)

target_compile_definitions(
  rl_congestion_control PUBLIC
  C10_USE_GLOG  # Fix glog compilation issues related to PyTorch
)
