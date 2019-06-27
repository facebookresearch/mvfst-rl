add_library(
  rl_congestion_control STATIC
  RLCongestionController.cpp
)

target_compile_options(
  rl_congestion_control
  PRIVATE
  ${_QUIC_COMMON_COMPILE_OPTIONS}
)

target_include_directories(
  rl_congestion_control PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  rl_congestion_control PUBLIC
  fizz::fizz
  fizz::fizz_test_support
  mvfst::mvfst_cc_algo
  mvfst::mvfst_client
  mvfst::mvfst_codec
  mvfst::mvfst_codec_decode
  mvfst::mvfst_codec_packet_number_cipher
  mvfst::mvfst_codec_pktbuilder
  mvfst::mvfst_codec_pktrebuilder
  mvfst::mvfst_codec_types
  mvfst::mvfst_constants
  mvfst::mvfst_exception
  mvfst::mvfst_flowcontrol
  mvfst::mvfst_handshake
  mvfst::mvfst_happyeyeballs
  mvfst::mvfst_looper
  mvfst::mvfst_loss
  mvfst::mvfst_qlogger
  mvfst::mvfst_server
  mvfst::mvfst_state_ack_handler
  mvfst::mvfst_state_functions
  mvfst::mvfst_state_machine
  mvfst::mvfst_state_pacing_functions
  mvfst::mvfst_state_qpr_functions
  mvfst::mvfst_state_simple_frame_functions
  mvfst::mvfst_state_stream
  mvfst::mvfst_state_stream_functions
  mvfst::mvfst_transport
)
