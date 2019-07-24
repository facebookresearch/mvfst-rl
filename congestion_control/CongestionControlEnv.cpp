#include "CongestionControlEnv.h"

#include <torch/torch.h>

namespace quic {

/// CongestionControlEnv impl

CongestionControlEnv::CongestionControlEnv(const Config& config, Callback* cob)
    : config_(config), cob_(CHECK_NOTNULL(cob)), observationTimeout_(this) {
  observationTimeout_.schedule(config.windowDuration);
}

void CongestionControlEnv::onUpdate(Observation&& observation) {
  // Update the observation with the last action taken
  observation.lastAction = lastAction_;

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
  lastAction_ = action;
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

    tensor_a[i][0] = obs.rttMinMs;
    tensor_a[i][1] = obs.rttStandingMs;
    tensor_a[i][2] = obs.lrttMs;
    tensor_a[i][3] = obs.srttMs;
    tensor_a[i][4] = obs.rttVarMs;
    tensor_a[i][5] = obs.delayMs;

    tensor_a[i][6] = obs.cwndBytes;
    tensor_a[i][7] = obs.bytesInFlight;
    tensor_a[i][8] = obs.writableBytes;
    tensor_a[i][9] = obs.bytesSent;
    tensor_a[i][10] = obs.bytesRecvd;
    tensor_a[i][11] = obs.bytesRetransmitted;

    tensor_a[i][12] = obs.ptoCount;
    tensor_a[i][13] = obs.totalPTODelta;
    tensor_a[i][14] = obs.rtxCount;
    tensor_a[i][15] = obs.timeoutBasedRtxCount;

    tensor_a[i][16] = obs.ackedBytes;
    tensor_a[i][17] = obs.ackedPackets;
    tensor_a[i][18] = obs.throughput;

    tensor_a[i][19] = obs.lostBytes;
    tensor_a[i][20] = obs.lostPackets;
    tensor_a[i][21] = obs.persistentCongestion;

    tensor_a[i][22] = obs.lastAction.cwndAction;
  }
}

}  // namespace quic
