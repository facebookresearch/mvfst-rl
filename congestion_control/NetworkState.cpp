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
      return "RTT_MIN";
    case Field::RTT_STANDING:
      return "RTT_STANDING";
    case Field::LRTT:
      return "LRTT";
    case Field::SRTT:
      return "SRTT";
    case Field::RTT_VAR:
      return "RTT_VAR";
    case Field::DELAY:
      return "DELAY";
    case Field::CWND:
      return "CWND";
    case Field::IN_FLIGHT:
      return "IN_FLIGHT";
    case Field::WRITABLE:
      return "WRITABLE";
    case Field::SENT:
      return "SENT";
    case Field::RECEIVED:
      return "RECEIVED";
    case Field::RETRANSMITTED:
      return "RETRANSMITTED";
    case Field::PTO_COUNT:
      return "PTO_COUNT";
    case Field::TOTAL_PTO_DELTA:
      return "TOTAL_PTO_DELTA";
    case Field::RTX_COUNT:
      return "RTX_COUNT";
    case Field::TIMEOUT_BASED_RTX_COUNT:
      return "TIMEOUT_BASED_RTX_COUNT";
    case Field::ACKED:
      return "ACKED";
    case Field::THROUGHPUT:
      return "THROUGHPUT";
    case Field::LOST:
      return "LOST";
    case Field::PERSISTENT_CONGESTION:
      return "PERSISTENT_CONGESTION";
    case Field::NUM_FIELDS:
      return "NUM_FIELDS";
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
