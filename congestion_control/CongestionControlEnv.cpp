#include "CongestionControlEnv.h"

#include <folly/Conv.h>
#include <quic/congestion_control/CongestionControlFunctions.h>
#include <torch/torch.h>

namespace quic {

using Observation = CongestionControlEnv::Observation;
using Field = CongestionControlEnv::Observation::Field;

/// CongestionControlEnv impl

CongestionControlEnv::CongestionControlEnv(const Config& cfg, Callback* cob,
                                           const QuicConnectionStateBase& conn)
    : cfg_(cfg),
      cob_(CHECK_NOTNULL(cob)),
      conn_(conn),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen),
      pastActions_(cfg.numPastActions, Action{0}),  // NOOP past actions
      evb_(folly::EventBaseManager::get()->getEventBase()),
      observationTimeout_(this, evb_) {
  if (cfg.aggregation == Config::Aggregation::TIME_WINDOW) {
    CHECK_GT(cfg.windowDuration.count(), 0);
    observationTimeout_.schedule(cfg.windowDuration);
  }
}

void CongestionControlEnv::onUpdate(Observation&& obs) {
  // Update the observation with the past actions taken
  obs.setPastActions(pastActions_);

  VLOG(2) << obs;

  observations_.emplace_back(std::move(obs));
  switch (cfg_.aggregation) {
    case Config::Aggregation::TIME_WINDOW:
      DCHECK(observationTimeout_.isScheduled());
      break;
    case Config::Aggregation::FIXED_WINDOW:
      if (observations_.size() == cfg_.windowSize) {
        onObservation(observations_);
        observations_.clear();
      }
      break;
  }
}

void CongestionControlEnv::onAction(const Action& action) {
  evb_->runImmediatelyOrRunInEventBaseThreadAndWait([this, action] {
    updateCwnd(action.cwndAction);
    cob_->onUpdate(cwndBytes_);

    // Keep track of past actions taken
    pastActions_.pop_front();
    pastActions_.push_back(action);
  });
}

void CongestionControlEnv::observationTimeoutExpired() noexcept {
  if (!observations_.empty()) {
    onObservation(observations_);
    observations_.clear();
  }
  observationTimeout_.schedule(cfg_.windowDuration);
}

void CongestionControlEnv::updateCwnd(const uint32_t actionIdx) {
  DCHECK_LT(actionIdx, cfg_.actions.size());
  const auto& op = cfg_.actions[actionIdx].first;
  const auto& val = cfg_.actions[actionIdx].second;

  switch (op) {
    case Config::ActionOp::NOOP:
      break;
    case Config::ActionOp::ADD:
      cwndBytes_ += val * conn_.udpSendPacketLen;
      break;
    case Config::ActionOp::SUB:
      cwndBytes_ -= val * conn_.udpSendPacketLen;
      break;
    case Config::ActionOp::MUL:
      cwndBytes_ = std::round(cwndBytes_ * val);
      break;
    case Config::ActionOp::DIV:
      cwndBytes_ = std::round(cwndBytes_ * 1.0 / val);
      break;
    default:
      LOG(FATAL) << "Unknown ActionOp";
      break;
  }

  cwndBytes_ = boundedCwnd(cwndBytes_, conn_.udpSendPacketLen,
                           conn_.transportSettings.maxCwndInMss,
                           conn_.transportSettings.minCwndInMss);
}

/// CongestionControlEnv::Observation impl

float Observation::reward(const std::vector<Observation>& observations,
                          const Config& cfg) {
  // Reward function is a combinaton of throughput, delay and lost bytes.
  // For throughput and delay, it makes sense to take the average, whereas
  // for loss, we compute the total bytes lost over these observations.
  float avgThroughput = 0.0;
  float avgDelay = 0.0;
  float maxDelay = 0.0;
  float totalLost = 0.0;
  for (const auto& obs : observations) {
    avgThroughput += obs[Field::THROUGHPUT];
    avgDelay += obs[Field::DELAY];
    maxDelay = std::max(maxDelay, obs[Field::DELAY]);
    totalLost += obs[Field::LOST];
  }
  avgThroughput /= observations.size();
  avgDelay /= observations.size();

  // Convert back to original scale by undoing the normalization. This brings
  // throughput and delay to a somewhat similar to scale, especially when
  // taking log. That isn't the case for lost bytes though, so it'll be
  // important to set the packetLossFactor to a very low value like 0.1.
  // But honestly, all of this is dogscience.
  float throughputBytesPerMs = avgThroughput * cfg.normBytes / cfg.normMs;
  float avgDelayMs = avgDelay * cfg.normMs;
  float maxDelayMs = maxDelay * cfg.normMs;
  float delayMs = (cfg.maxDelayInReward ? maxDelayMs : avgDelayMs);
  float lostBytes = totalLost * cfg.normBytes;

  // TODO (viswanath): Differences in reward scale based on network condition.
  float reward = cfg.throughputFactor * std::log(throughputBytesPerMs) -
                 cfg.delayFactor * std::log(1 + delayMs) -
                 cfg.packetLossFactor * std::log(1 + lostBytes);
  VLOG(1) << "Num observations = " << observations.size()
          << ", avg throughput = " << throughputBytesPerMs
          << " bytes/ms, avg delay = " << avgDelayMs
          << " ms, max delay = " << maxDelayMs
          << " ms total bytes lost = " << lostBytes << ", reward = " << reward;
  return reward;
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
  if (observations.empty()) {
    tensor.resize_({0});
    return;
  }

  tensor.resize_({observations.size(), observations[0].size()});
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
    default:
      return "Field " + folly::to<std::string>(static_cast<int>(field));
  }
  __builtin_unreachable();
}

std::ostream& operator<<(std::ostream& os, const Observation& obs) {
  os << "Observation (" << obs.size() << " fields):" << std::endl;
  for (size_t i = 0; i < obs.size(); ++i) {
    os << i << ". " << Observation::fieldToString(i) << " = " << obs[i]
       << std::endl;
  }
  return os;
}

}  // namespace quic
