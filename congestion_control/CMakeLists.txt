add_library(
  rl_congestion_control STATIC
  RLCongestionController.cpp
  CongestionControlEnv.cpp
  CongestionControlRPCEnv.cpp
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
  rpcenv_pb
)
