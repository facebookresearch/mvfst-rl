#include "CongestionControlEnv.h"

#include <torch/torch.h>

namespace quic {

using Observation = CongestionControlEnv::Observation;
using Field = CongestionControlEnv::Observation::Field;

/// CongestionControlEnv impl

CongestionControlEnv::CongestionControlEnv(const Config& config, Callback* cob)
    : config_(config), cob_(CHECK_NOTNULL(cob)), observationTimeout_(this) {
  observationTimeout_.schedule(config.windowDuration);
}

void CongestionControlEnv::onUpdate(Observation&& obs) {
  // Update the observation with the last action taken
  obs[Field::PREV_CWND_ACTION] = prevAction_.cwndAction;

  VLOG(4) << obs;

  observations_.emplace_back(std::move(obs));
  switch (config_.aggregation) {
    case Aggregation::TIME_WINDOW:
      DCHECK(observationTimeout_.isScheduled());
      break;
    case Aggregation::FIXED_WINDOW:
      if (observations_.size() == config_.windowSize) {
        onObservation(observations_);
        observations_.clear();
      }
      break;
  }
}

void CongestionControlEnv::onAction(const Action& action) {
  // TODO (viswanath): impl, callback
  prevAction_ = action;
}

void CongestionControlEnv::onReset() { cob_->onReset(); }

void CongestionControlEnv::observationTimeoutExpired() noexcept {
  if (!observations_.empty()) {
    onObservation(observations_);
    observations_.clear();
  }
  observationTimeout_.schedule(config_.windowDuration);
}

/// CongestionControlEnv::Observation impl

float Observation::reward(const std::vector<Observation>& observations) {
  // TODO (viswanath): impl
  return 0;
}

torch::Tensor Observation::toTensor() const {
  torch::Tensor tensor;
  toTensor(tensor);
  return tensor;
}

void Observation::toTensor(torch::Tensor& tensor) const {
  toTensor({*this}, tensor);
}

torch::Tensor Observation::toTensor(
    const std::vector<Observation>& observations) {
  torch::Tensor tensor;
  toTensor(observations, tensor);
  return tensor;
}

void Observation::toTensor(const std::vector<Observation>& observations,
                           torch::Tensor& tensor) {
  tensor.resize_({observations.size(), Observation::kNumFields});
  auto tensor_a = tensor.accessor<float, 2>();
  for (int i = 0; i < tensor_a.size(0); ++i) {
    for (int j = 0; j < tensor_a.size(1); ++j) {
      tensor_a[i][j] = observations[i][j];
    }
  }
}

std::string Observation::fieldToString(const uint16_t field) {
  return fieldToString(static_cast<Field>(field));
}

std::string Observation::fieldToString(const Field field) {
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
    case Field::PREV_CWND_ACTION:
      return "PREV_CWND_ACTION";
    case Field::NUM_FIELDS:
      return "NUM_FIELDS";
  }
}

std::ostream& operator<<(std::ostream& os, const Observation& obs) {
  os << "Observation (" << obs.size() << " fields):" << std::endl;
  for (int i = 0; i < obs.size(); ++i) {
    os << i << ". " << Observation::fieldToString(i) << " = " << obs[i]
       << std::endl;
  }
  return os;
}

}  // namespace quic
