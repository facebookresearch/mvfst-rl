#include "CongestionControlEnv.h"

#include <torch/torch.h>

namespace quic {

using Field = CongestionControlEnv::Observation::Field;

/// CongestionControlEnv impl

CongestionControlEnv::CongestionControlEnv(const Config& config, Callback* cob)
    : config_(config), cob_(CHECK_NOTNULL(cob)), observationTimeout_(this) {
  observationTimeout_.schedule(config.windowDuration);
}

void CongestionControlEnv::onUpdate(Observation&& obs) {
  // Update the observation with the last action taken
  obs[Field::PREV_CWND_ACTION] = prevAction_.cwndAction;

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

float CongestionControlEnv::Observation::reward(
    const std::vector<Observation>& observations) {
  // TODO (viswanath): impl
  return 0;
}

torch::Tensor CongestionControlEnv::Observation::toTensor() const {
  torch::Tensor tensor;
  toTensor(tensor);
  return tensor;
}

void CongestionControlEnv::Observation::toTensor(torch::Tensor& tensor) const {
  toTensor({*this}, tensor);
}

torch::Tensor CongestionControlEnv::Observation::toTensor(
    const std::vector<Observation>& observations) {
  torch::Tensor tensor;
  toTensor(observations, tensor);
  return tensor;
}

void CongestionControlEnv::Observation::toTensor(
    const std::vector<Observation>& observations, torch::Tensor& tensor) {
  tensor.resize_({observations.size(), Observation::NUM_FIELDS});
  auto tensor_a = tensor.accessor<float, 2>();
  for (int i = 0; i < tensor_a.size(0); ++i) {
    for (int j = 0; j < tensor_a.size(1); ++j) {
      tensor_a[i][j] = observations[i][j];
    }
  }
}

}  // namespace quic
