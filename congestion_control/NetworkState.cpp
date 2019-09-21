/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include "NetworkState.h"

#include <glog/logging.h>

namespace quic {

torch::Tensor NetworkState::toTensor() const {
  torch::Tensor tensor = torch::empty({0}, torch::kFloat32);
  toTensor(tensor);
  return tensor;
}

void NetworkState::toTensor(torch::Tensor& tensor) const {
  toTensor({*this}, tensor);
}

torch::Tensor NetworkState::toTensor(const std::vector<NetworkState>& states) {
  torch::Tensor tensor = torch::empty({0}, torch::kFloat32);
  toTensor(states, tensor);
  return tensor;
}

void NetworkState::toTensor(const std::vector<NetworkState>& states,
                            torch::Tensor& tensor) {
  if (states.empty()) {
    tensor.resize_({0});
    return;
  }

  tensor.resize_({states.size(), states[0].size()});
  auto tensor_a = tensor.accessor<float, 2>();
  for (int i = 0; i < tensor_a.size(0); ++i) {
    for (int j = 0; j < tensor_a.size(1); ++j) {
      tensor_a[i][j] = states[i][j];
    }
  }
}

std::vector<NetworkState> NetworkState::fromTensor(
    const torch::Tensor& tensor) {
  CHECK_EQ(tensor.dim(), 2);
  CHECK_EQ(tensor.sizes()[1], kNumFields);

  std::vector<NetworkState> states;
  auto tensor_a = tensor.accessor<float, 2>();
  for (int i = 0; i < tensor_a.size(0); ++i) {
    NetworkState state;
    for (int j = 0; j < tensor_a.size(1); ++j) {
      state[j] = tensor_a[i][j];
    }
    states.push_back(std::move(state));
  }
  return states;
}

std::string NetworkState::fieldToString(const uint16_t field) {
  return fieldToString(static_cast<Field>(field));
}

std::string NetworkState::fieldToString(const Field field) {
  switch (field) {
    case Field::RTT_MIN:
      return "rtt_min";
    case Field::RTT_STANDING:
      return "rtt_standing";
    case Field::LRTT:
      return "lrtt";
    case Field::SRTT:
      return "srtt";
    case Field::RTT_VAR:
      return "rtt_var";
    case Field::DELAY:
      return "delay";
    case Field::CWND:
      return "cwnd";
    case Field::IN_FLIGHT:
      return "in_flight";
    case Field::WRITABLE:
      return "writable";
    case Field::SENT:
      return "sent";
    case Field::RECEIVED:
      return "received";
    case Field::RETRANSMITTED:
      return "retransmitted";
    case Field::PTO_COUNT:
      return "pto_count";
    case Field::TOTAL_PTO_DELTA:
      return "total_pto_delta";
    case Field::RTX_COUNT:
      return "rtx_count";
    case Field::TIMEOUT_BASED_RTX_COUNT:
      return "timeout_based_rtx_count";
    case Field::ACKED:
      return "acked";
    case Field::THROUGHPUT:
      return "throughput";
    case Field::LOST:
      return "lost";
    case Field::PERSISTENT_CONGESTION:
      return "persistent_congestion";
    case Field::NUM_FIELDS:
      return "num_fields";
    default:
      LOG(FATAL) << "Unknown field";
      break;
  }
  __builtin_unreachable();
}

std::ostream& operator<<(std::ostream& os, const NetworkState& state) {
  os << "NetworkState (" << state.size() << " fields): ";
  for (size_t i = 0; i < state.size(); ++i) {
    os << NetworkState::fieldToString(i) << "=" << state[i] << " ";
  }
  return os;
}

}  // namespace quic
