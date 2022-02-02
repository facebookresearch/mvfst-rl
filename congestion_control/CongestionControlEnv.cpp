/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include "CongestionControlEnv.h"

#include <fstream>

#include <folly/Conv.h>
#include <quic/congestion_control/CongestionControlFunctions.h>

namespace quic {

using Field = NetworkState::Field;

static const float kBytesToMB = 1e-6;

/// CongestionControlEnv impl

CongestionControlEnv::CongestionControlEnv(const Config &cfg, Callback *cob,
                                           const QuicConnectionStateBase &conn)
    : cfg_(cfg), conn_(conn), cob_(CHECK_NOTNULL(cob)),
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

  if (!cfg_.statsFile.empty()) {
    std::ofstream outfile;
    outfile.open(cfg_.statsFile);
    outfile << "episode_step avg_noisy_rtt_no_delay" << std::endl;
  }
}

void CongestionControlEnv::onAction(const Action &action) {
  evb_->runImmediatelyOrRunInEventBaseThreadAndWait([this, action] {
    updateCwnd(action.cwndAction);
    cob_->onUpdate(cwndBytes_);

    // Update history
    history_.pop_front();
    history_.emplace_back(
        action,
        cwndBytes_ / static_cast<float>(conn_.transportSettings.maxCwndInMss *
                                        conn_.udpSendPacketLen));

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
    VLOG(1) << __func__ << ": for jobCount= " << cfg_.jobCount << ", after "
            << rewardCount_
            << " steps, avg reward = " << (rewardSum_ / rewardCount_);
  }
  if (rewardCount_ % 50 == 0 && !cfg_.statsFile.empty()) {
    // Save some stats every 50 steps.
    std::ofstream outfile;
    outfile.open(cfg_.statsFile, std::ofstream::app);
    outfile << rewardCount_ << " " << avgNoisyRTTNoDelay_ << std::endl;
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
      Field::RTT_MIN,
      // Field::RTT_STANDING, Field::LRTT,     Field::LDRTT,
      // Field::SRTT,    Field::RTT_VAR,
      Field::NOISY_RTT_WITH_DELAY,
      Field::DELAY, Field::ACK_DELAY, Field::CWND, Field::THROUGHPUT_FROM_CWND,
      // Field::IN_FLIGHT,    Field::WRITABLE,
      Field::THROUGHPUT,
      // Field::THROUGHPUT_LOG_RATIO,
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
  float avgAckDelay = 0.0;
  float totalLost = 0.0;
  float minRtt = std::numeric_limits<float>::max();
  static uint64_t prevCwndBytes = 10;
  avgNoisyRTTNoDelay_ = 0.0;
  for (const auto &state : states) {
    avgThroughput += state[Field::THROUGHPUT];
    avgDelay += state[Field::DELAY];
    maxDelay = std::max(maxDelay, state[Field::DELAY]);
    avgAckDelay += state[Field::ACK_DELAY];
    // totalLost += state[Field::LOST];
    minRtt = std::min(minRtt, state[Field::RTT_MIN]);
    avgNoisyRTTNoDelay_ += state[Field::NOISY_RTT_WITH_DELAY] - state[Field::ACK_DELAY];
  }
  avgThroughput /= states.size();
  avgDelay /= states.size();
  avgAckDelay /= states.size();
  avgNoisyRTTNoDelay_ *= normMs() / states.size();

  // Undo normalization and convert to MB/sec for throughput and ms for
  // delay.
  const float throughputMBps = avgThroughput * normBytes() * kBytesToMB;
  const float avgDelayMs = avgDelay * normMs();
  const float maxDelayMs = maxDelay * normMs();
  float delayMs = (cfg_.maxDelayInReward ? maxDelayMs : avgDelayMs);
  delayMs = std::max(0.f, delayMs - cfg_.delayOffset);
  const float avgAckDelayMs = avgAckDelay * normMs();
  const float lostMbits = totalLost * normBytes() * kBytesToMB;

  // The target CWND is computed from the theoretical max bandwidth available on
  // the uplink, as well as the theoretical min achievable RTT (which is
  // augmented with the observed ACK delay). This gives the formula:
  //  target cwnd (in bytes) = bandwidth * (rtt + ack_delay)
  const float targetCwndBytes =
      cfg_.uplinkBandwidth / kBytesToMB *
      (cfg_.baseRTT + avgAckDelayMs) / 1000.f;

  float reward = 0.f;

  switch (cfg_.rewardFormula) {

  case Config::RewardFormula::LINEAR:
    reward = cfg_.throughputFactor * throughputMBps -
             cfg_.delayFactor * delayMs - cfg_.packetLossFactor * lostMbits;
    break;

  case Config::RewardFormula::LOG_RATIO:
    reward =
        cfg_.throughputFactor * log(cfg_.throughputLogOffset + throughputMBps) -
        cfg_.delayFactor * log(cfg_.delayLogOffset + delayMs) -
        cfg_.packetLossFactor * log(cfg_.packetLossLogOffset + lostMbits);
    break;

  case Config::RewardFormula::MIN_THROUGHPUT: {
    float rewardThroughput = 1.f;
    if (throughputMBps < cfg_.minThroughputRatio * cfg_.uplinkBandwidth) {
      rewardThroughput =
          1.f - 2.f / (1.f + throughputMBps / cfg_.uplinkBandwidth);
    }
    // TODO this formula is actually wrong because the size of a "queue packet"
    // is not the same as an UDP packet  (1500 vs. 1252).
    // This reward formula will probably be removed at some point though, so
    // it is not a big deal.
    const float nPacketsInQueue = delayMs / 1000.f * cfg_.uplinkBandwidth /
                                  kBytesToMB /
                                  static_cast<float>(conn_.udpSendPacketLen);
    const float rewardDelay =
        log(cfg_.nPacketsOffset) - log(cfg_.nPacketsOffset + nPacketsInQueue);
    reward = cfg_.throughputFactor * rewardThroughput +
             cfg_.delayFactor * rewardDelay;
    break;
  }

  case Config::RewardFormula::TARGET_CWND:
    reward = cfg_.throughputFactor *
             (1.f - std::min(1.f, abs(1.f - cwndBytes_ / targetCwndBytes)));
    if (cwndBytes_ > targetCwndBytes) {
      reward -= cfg_.delayFactor *
                std::min(1.f, delayMs / std::max(1.f, minRtt * normMs()));
    }
    break;

  case Config::RewardFormula::TARGET_CWND_SHAPED: {
    float rewardThroughput = -1.f;
    const float diff = abs(cwndBytes_ - targetCwndBytes);
    if (diff < 1.f or diff < abs(prevCwndBytes - targetCwndBytes)) {
      rewardThroughput = 1.f;
    }
    prevCwndBytes = cwndBytes_;
    reward = cfg_.throughputFactor * rewardThroughput;
    break;
  }

  case Config::RewardFormula::HIGHER_IS_BETTER: {
    float rewardThroughput = -1.f;
    if (cwndBytes_ > prevCwndBytes || cwndBytes_ >= targetCwndBytes) {
      rewardThroughput = 1.f;
    }
    prevCwndBytes = cwndBytes_;
    reward = cfg_.throughputFactor * rewardThroughput;
    break;
  }

  case Config::RewardFormula::ABOVE_CWND: {
    const float rewardThroughput =
        cwndBytes_ >= cfg_.minThroughputRatio * targetCwndBytes ? 1.f : 0.f;
    const float rewardDelay = -log(cfg_.delayLogOffset + delayMs);
    reward = cfg_.throughputFactor * rewardThroughput +
             cfg_.delayFactor * rewardDelay;
    break;
  }

  case Config::RewardFormula::CWND_RANGE:
    reward = cfg_.throughputFactor *
             (cwndBytes_ >= cfg_.minThroughputRatio * targetCwndBytes &&
                      cwndBytes_ <= cfg_.maxThroughputRatio * targetCwndBytes
                  ? 1.f
                  : 0.f);
    break;

  case Config::RewardFormula::CWND_RANGE_SOFT: {
    const float low = cfg_.minThroughputRatio * targetCwndBytes;
    const float high = cfg_.maxThroughputRatio * targetCwndBytes;
    float rewardThroughput = 0.f;
    if (cwndBytes_ >= low && cwndBytes_ <= high) {
      const float mid = (low + high) * 0.5f;
      rewardThroughput =
          high > low
              // The `std::max()` is just to account for potential precision
              // issues, to avoid negative rewards.
              ? std::max(0.f, 1.f - 2.f * abs(cwndBytes_ - mid) / (high - low))
              : 1.f;
    }
    reward = cfg_.throughputFactor * rewardThroughput;
    break;
  }

  case Config::RewardFormula::CWND_TRADEOFF: {
    const float rewardThroughput = std::min(1.f, cwndBytes_ / targetCwndBytes);
    float rewardDelay = 1.f;
    if (cwndBytes_ > targetCwndBytes) {
      const float cwndScaleRTT =
          cfg_.baseRTT / 1000.f * cfg_.uplinkBandwidth / kBytesToMB;
      const float rewardRTT =
          1.f - (cwndBytes_ - targetCwndBytes) / cwndScaleRTT;
      const float rewardQueue =
          1.f - (cwndBytes_ - targetCwndBytes -
                 cfg_.uplinkQueueSizeBytes * cfg_.uplinkQueueMaxFillRatio) /
                    (cfg_.uplinkQueueSizeBytes *
                     (1.f - cfg_.uplinkQueueMaxFillRatio));
      rewardDelay =
          std::min(1.f, std::max(0.f, std::min(rewardRTT, rewardQueue)));
    }
    reward = (cfg_.throughputFactor * 2.f * (rewardThroughput - 0.5f) +
              cfg_.delayFactor * 2.f * (rewardDelay - 0.5f)) /
             (cfg_.throughputFactor + cfg_.delayFactor);

    break;
  }

  case Config::RewardFormula::BELOW_TARGET_CWND: {
    const float ratioTargetCwnd = targetCwndBytes * cfg_.maxThroughputRatio;
    if (cwndBytes_ > ratioTargetCwnd) {
      if (targetCwndBytes > ratioTargetCwnd) {
        reward = 1.f - std::min(1.f, (cwndBytes_ - ratioTargetCwnd) /
                                         (targetCwndBytes - ratioTargetCwnd)) *
                           (1 + cfg_.delayFactor);
      } else {
        reward = 0.f;
      }
    } else {
      reward = pow(cwndBytes_ / ratioTargetCwnd, cfg_.throughputFactor);
    }
    break;
  }

  default:
    LOG(FATAL) << "Unknown rewardFormula";
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
  // Total dim = flattened state dim + history dim
  // (must be kept in synch with `utils.get_observation_length()`)
  // Note that when useStateSummary is true, `states.size()` is equal to 5
  // (corresponding to the five aggretation statistics).
  uint32_t historyDim = cfg_.actions.size() + 1;
  uint32_t dim = states.size() * states[0].size() + history.size() * historyDim;

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

  // Scale final observation.
  tensor *= cfg_.obsScaling;

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
