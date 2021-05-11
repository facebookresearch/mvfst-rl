/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <chrono>

#include <quic/congestion_control/Bbr.h>

#include "Utils.h"

namespace quic {

using namespace std::chrono_literals;

// Bandwidth estimates must be computed over a window spanning at least this
// duration (to make these estimates more stable).
constexpr std::chrono::microseconds kBandwidthWindowMinDuration{100'000us};

/*
  Bandwidth estimator based on ACK packets.

  At high level the logic is as follows:
    - keep a rolling window of the last K ACK packets
    - estimate bandwdith as total #bytes acknowledged in these packets, divided
      by the window duration

  This class re-uses the same API as the bandwidth sampler from BBR so that it
  can be used as a "plug'n play" replacement.
*/
class RLBandwidthSampler : public BbrCongestionController::BandwidthSampler {
public:
  explicit RLBandwidthSampler(QuicConnectionStateBase &conn);

  Bandwidth getBandwidth() const noexcept override;

  // NB: this class actually ignores `rttCounter` as all computations are based
  // on timings included in `AckEvent`.
  void onPacketAcked(const CongestionController::AckEvent &,
                     uint64_t rttCounter) override;

  // For now we ignore app-limited mode.
  void onAppLimited() noexcept override {}
  bool isAppLimited() const noexcept override { return false; }

private:
  // Return the index in the rolling window corresponding to the last ACK
  // event that was received.
  uint64_t getPreviousIdx() const noexcept {
    return ackIdx_ > 0 ? ackIdx_ - 1 : ackBytes_.size() - 1;
  }

  QuicConnectionStateBase &conn_;

  // Rolling windows of (1) number of acked bytes, and (2) associated
  // timestamps (corresponding to the time each ACK was received).
  quic::utils::vector<uint64_t> ackBytes_;
  quic::utils::vector<TimePoint> ackTimes_;

  // We enforce a minimum window duration to avoid problematic situations where
  // several ACK events may be processed at (almost) the same time due to
  // network hiccups. Without a minimum duration, this could lead to an
  // unexpectedly high bandwidth estimate.
  std::chrono::microseconds minWindowDuration_{kBandwidthWindowMinDuration};

  // The minimum window duration above is translated into a minimum interval
  // between events stored in the rolling window (events too close to each other
  // are combined so as to respect this constraint).
  std::chrono::microseconds minIntervalBetweenAcks_;

  TimePoint lastEntryInitialTime_; // initial timestamp of last entry in window
  uint64_t totalAckBytes_{0};      // sum of acked bytes within window
  uint64_t ackIdx_{0};             // current index in rolling window
  bool gotFirstAck_{false};        // whether we have received the first ACK
};

} // namespace quic