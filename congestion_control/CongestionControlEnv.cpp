/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include "CongestionControlEnv.h"

#include <folly/Conv.h>
#include <quic/congestion_control/CongestionControlFunctions.h>

namespace quic {

using Field = NetworkState::Field;

static const float kBytesToMB = 1e-6;

/// CongestionControlEnv impl

CongestionControlEnv::CongestionControlEnv(const Config &cfg, Callback *cob,
                                           const QuicConnectionStateBase &conn)
    : cfg_(cfg), cob_(CHECK_NOTNULL(cob)), conn_(conn),
      evb_(folly::EventBaseManager::get()->getEventBase()),
      observationTimeout_(this, evb_),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen),
      rewardCount_(0), rewardSum_(0.f) {
  // Initialize history with no-op past actions
  History noopHistory(Action{0}, cwndBytes_ / normBytes());
  history_.resize(cfg.historySize, noopHistory);

  if (cfg.aggregation == Config::Aggregation::TIME_WINDOW) {
    CHECK_GT(cfg.windowDuration.count(), 0);
    observationTimeout_.schedule(cfg.windowDuration);
  }
}

void CongestionControlEnv::onAction(const Action &action) {
  evb_->runImmediatelyOrRunInEventBaseThreadAndWait([this, action] {
    updateCwnd(action.cwndAction);
    cob_->onUpdate(cwndBytes_);

    // Update history
    history_.pop_front();
    history_.emplace_back(action, cwndBytes_ / normBytes());

    const auto &elapsed = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - lastObservationTime_);
    VLOG(1) << "Action updated (cwndAction=" << action.cwndAction
            << ", cwnd=" << cwndBytes_ / conn_.udpSendPacketLen
            << "), policy elapsed time = " << elapsed.count() << " ms";
  });
}

uint64_t CongestionControlEnv::getUpdatedCwndBytes(uint64_t currentCwndBytes,
                                                   uint32_t actionIdx) const {
  DCHECK_LT(actionIdx, cfg_.actions.size());
  const auto &op = cfg_.actions[actionIdx].first;
  const auto &val = cfg_.actions[actionIdx].second;
  const auto &valBytes = val * conn_.udpSendPacketLen;

  switch (op) {
  case Config::ActionOp::NOOP:
    break;
  case Config::ActionOp::ADD:
    currentCwndBytes += valBytes;
    break;
  case Config::ActionOp::SUB:
    currentCwndBytes =
        (currentCwndBytes >= valBytes) ? (currentCwndBytes - valBytes) : 0;
    break;
  case Config::ActionOp::MUL:
    currentCwndBytes = std::round(currentCwndBytes * val);
    break;
  case Config::ActionOp::DIV:
    currentCwndBytes = std::round(currentCwndBytes * 1.0 / val);
    break;
  default:
    LOG(FATAL) << "Unknown ActionOp";
    break;
  }

  return boundedCwnd(currentCwndBytes, conn_.udpSendPacketLen,
                     conn_.transportSettings.maxCwndInMss,
                     conn_.transportSettings.minCwndInMss);
}

void CongestionControlEnv::onNetworkState(NetworkState &&state) {
  VLOG(3) << __func__ << ": " << state;

  states_.push_back(std::move(state));

  switch (cfg_.aggregation) {
  case Config::Aggregation::TIME_WINDOW:
    DCHECK(observationTimeout_.isScheduled());
    break;
  case Config::Aggregation::FIXED_WINDOW:
    if (states_.size() == cfg_.windowSize) {
      handleStates();
    }
    break;
  default:
    LOG(FATAL) << "Unknown aggregation type";
    break;
  }
}

void CongestionControlEnv::observationTimeoutExpired() noexcept {
  handleStates();
  observationTimeout_.schedule(cfg_.windowDuration);
}

void CongestionControlEnv::handleStates() {
  if (states_.empty()) {
    return;
  }

  // Compute reward based on original states
  const float reward = computeReward(states_);

  ++rewardCount_;
  rewardSum_ += reward;
  if (rewardCount_ % 10 == 0) {
    VLOG(1) << __func__ << ": for jobId= " << cfg_.jobId
            << ", after " << rewardCount_
            << " steps, avg reward = " << (rewardSum_ / rewardCount_);
  }

  Observation obs(cfg_);
  obs.states = useStateSummary() ? stateSummary(states_) : std::move(states_);
  states_.clear();
  std::copy(history_.begin(), history_.end(), std::back_inserter(obs.history));

  VLOG(2) << __func__ << ' ' << obs;

  lastObservationTime_ = std::chrono::steady_clock::now();
  onObservation(std::move(obs), reward);
}

quic::utils::vector<NetworkState> CongestionControlEnv::stateSummary(
    const quic::utils::vector<NetworkState> &states) {
  int dim = 0;
  bool keepdim = true;
  // Bassel's correction on stddev only when defined to avoid NaNs.
  bool unbiased = (states.size() > 1);

  NetworkState::toTensor(states, summaryTensor_);
  const auto &sum = torch::sum(summaryTensor_, dim, keepdim);
  const auto &std_mean =
      torch::std_mean(summaryTensor_, dim, unbiased, keepdim);
  const auto &min = torch::amin(summaryTensor_, dim, keepdim);
  const auto &max = torch::amax(summaryTensor_, dim, keepdim);
  // If these statistics are modified / re-ordered, make sure to also update
  // the corresponding `OFFSET_*` constants in state.py.
  const auto &summary = torch::cat(
      {sum, std::get<1>(std_mean), std::get<0>(std_mean), min, max}, dim);
  auto summaryStates = NetworkState::fromTensor(summary);

  // Certain stats for some fields don't make sense such as sum over
  // RTT from ACKs. Zero-out them.
  static const quic::utils::vector<Field> invalidSumFields = {
      Field::RTT_MIN, Field::RTT_STANDING, Field::LRTT,
      Field::SRTT,    Field::RTT_VAR,      Field::DELAY,
      Field::CWND,    Field::IN_FLIGHT,    Field::WRITABLE,
  };
  for (const Field field : invalidSumFields) {
    summaryStates[0][field] = 0.0;
  }

  static const quic::utils::vector<std::string> keys = {
      "Sum", "Mean", "Std", "Min", "Max",
  };
  VLOG(2) << "State summary: ";
  for (size_t i = 0; i < summaryStates.size(); ++i) {
    VLOG(2) << keys[i] << ": " << summaryStates[i];
  }

  return summaryStates;
}

float CongestionControlEnv::computeReward(
    const quic::utils::vector<NetworkState> &states) const {
  // Reward function is a combinaton of throughput, delay and lost bytes.
  // For throughput and delay, it makes sense to take the average, whereas
  // for loss, we compute the total bytes lost over these states.
  float avgThroughput = 0.0;
  float avgDelay = 0.0;
  float maxDelay = 0.0;
  float totalLost = 0.0;
  for (const auto &state : states) {
    avgThroughput += state[Field::THROUGHPUT];
    avgDelay += state[Field::DELAY];
    maxDelay = std::max(maxDelay, state[Field::DELAY]);
    totalLost += state[Field::LOST];
  }
  avgThroughput /= states.size();
  avgDelay /= states.size();

  // Undo normalization and convert to MB/sec for throughput and ms for
  // delay.
  float throughputMBps = avgThroughput * normBytes() * kBytesToMB;
  float avgDelayMs = avgDelay * normMs();
  float maxDelayMs = maxDelay * normMs();
  float delayMs = (cfg_.maxDelayInReward ? maxDelayMs : avgDelayMs);
  float lostMbits = totalLost * normBytes() * kBytesToMB;

  float reward = 0.f;
  if (cfg_.rewardLogRatio) {
    reward =
        cfg_.throughputFactor * log(cfg_.throughputLogOffset + throughputMBps) -
        cfg_.delayFactor * log(cfg_.delayLogOffset + delayMs) -
        cfg_.packetLossFactor * log(cfg_.packetLossLogOffset + lostMbits);
  } else {
    reward = cfg_.throughputFactor * throughputMBps -
             cfg_.delayFactor * delayMs - cfg_.packetLossFactor * lostMbits;
  }
  VLOG(1) << "Num states = " << states.size()
          << " avg throughput = " << throughputMBps
          << " MB/sec, avg delay = " << avgDelayMs
          << " ms, max delay = " << maxDelayMs
          << " ms, total Mb lost = " << lostMbits << ", reward = " << reward;
  return reward;
}

void CongestionControlEnv::updateCwnd(const uint32_t actionIdx) {
  cwndBytes_ = getUpdatedCwndBytes(cwndBytes_, actionIdx);
}

/// CongestionControlEnv::Observation impl

torch::Tensor CongestionControlEnv::Observation::toTensor() const {
  torch::Tensor tensor = torch::empty({0}, torch::kFloat32);
  toTensor(tensor);
  return tensor;
}

void CongestionControlEnv::Observation::toTensor(torch::Tensor &tensor) const {
  if (states.empty()) {
    tensor.resize_({0});
    return;
  }

  CHECK_EQ(history.size(), cfg_.historySize);

  // Dim per history = len(one-hot actions) + 1 (cwnd).
  // Total dim = flattened state dim + history dim + 1 (job ID)
  uint32_t historyDim = cfg_.actions.size() + 1;
  uint32_t dim = states.size() * states[0].size() + history.size() * historyDim + 1;

  tensor.resize_({dim});
  auto tensor_a = tensor.accessor<float, 1>();
  int x = 0;

  // Serialize states
  for (const auto &state : states) {
    for (size_t i = 0; i < state.size(); ++i) {
      tensor_a[x++] = state[i];
    }
  }

  // Serialize history
  for (const auto &h : history) {
    for (size_t i = 0; i < cfg_.actions.size(); ++i) {
      tensor_a[x++] = (h.action.cwndAction == i);
    }
    tensor_a[x++] = h.cwnd;
  }

  // Append the job ID (IMPORTANT: it must remain at the end of the tensor)
  tensor_a[x++] = cfg_.jobId;

  CHECK_EQ(x, dim);
}

std::ostream &operator<<(std::ostream &os,
                         const CongestionControlEnv::Observation &obs) {
  os << "Observation (" << obs.states.size() << " states, "
     << obs.history.size() << " history):" << std::endl;
  for (const auto &state : obs.states) {
    os << state << std::endl;
  }
  for (const auto &history : obs.history) {
    os << history << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const CongestionControlEnv::History &history) {
  os << "History: action=" << history.action.cwndAction
     << " cwnd=" << history.cwnd;
  return os;
}

} // namespace quic
