#include "CongestionControlEnv.h"

#include <torch/torch.h>

namespace quic {

/// CongestionControlEnv impl

CongestionControlEnv::CongestionControlEnv(const Config& config, Callback* cob)
    : config_(config), cob_(CHECK_NOTNULL(cob)), observationTimeout_(this) {
  observationTimeout_.schedule(config.windowDuration);
}

void CongestionControlEnv::onUpdate(Observation&& observation) {
  observations_.emplace_back(std::move(observation));
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
}

void CongestionControlEnv::observationTimeoutExpired() noexcept {
  if (!observations_.empty()) {
    onObservation(observations_);
    observations_.clear();
  }
  observationTimeout_.schedule(config_.windowDuration);
}

/// CongestionControlEnv::Observation impl

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
  tensor.resize_({observations.size(), Observation::DIMS});
  auto tensor_a = tensor.accessor<float, 2>();
  for (int i = 0; i < tensor_a.size(0); ++i) {
    const auto& obs = observations[i];
    tensor_a[i][0] = obs.rtt;
    tensor_a[i][1] = obs.cwndBytes;
    // TODO (viswanath): Add more stuff
  }
}

}  // namespace quic
