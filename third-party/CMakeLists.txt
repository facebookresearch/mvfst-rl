# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Exports helper static libs from third-party deps

if(NOT INFERENCE_ONLY)
  # Lib for generated libtorchbeast protobuf files.
  # Somewhat follows from torchbeast/setup.py
  add_library(
    rpcenv_pb STATIC
    torchbeast/libtorchbeast/rpcenv.pb.cc
    torchbeast/libtorchbeast/rpcenv.grpc.pb.cc
  )

  target_compile_options(
    rpcenv_pb
    PRIVATE
    ${_QUIC_COMMON_COMPILE_OPTIONS}
    -Wno-overloaded-virtual  # gRPC is noisy
  )

  target_link_libraries(
    rpcenv_pb PUBLIC
    grpc
    grpc++
    gpr
    address_sorting
    protobuf
  )
endif()

# A single interface lib that links together all mvfst libs and dependencies
# for ease of use.
add_library(mvfst INTERFACE)

target_link_libraries(
  mvfst INTERFACE
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
