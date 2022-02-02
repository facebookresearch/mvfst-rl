/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/

#include "RLBandwidthSampler.h"

namespace quic {

using namespace std::chrono;

RLBandwidthSampler::RLBandwidthSampler(QuicConnectionStateBase &conn, const CongestionControlEnv::Config &cfg)
    : conn_(conn), ackBytes_(kBandwidthWindowLength, 0),
      ackTimes_(kBandwidthWindowLength, Clock::now()),
      minWindowDuration_(cfg.bandwidthMinWindowDuration),
      lastEntryInitialTime_(Clock::now()) {
  // Compute the min interval required to respect the min window duration.
  const uint64_t minWin = minWindowDuration_.count();
  const uint64_t winSize = ackBytes_.size();
  // Ceiling of integer division, see e.g.
  // https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
  minIntervalBetweenAcks_ =
      std::chrono::microseconds(minWin / winSize + (minWin % winSize != 0));
  VLOG(10) << __func__ << ": minWindowDuration = " << minWindowDuration_.count()
           << ", window size = " << ackBytes_.size()
           << ", minIntervalBetweenAcks = " << minIntervalBetweenAcks_.count();
}

Bandwidth RLBandwidthSampler::getBandwidth() const noexcept {
  // Compute current window duration, lower bounded by its min allowed value.
  const uint64_t previousIdx = getPreviousIdx();
  std::chrono::microseconds windowDuration =
      duration_cast<microseconds>(ackTimes_[previousIdx] - ackTimes_[ackIdx_]);
  DCHECK(windowDuration.count() >= 0);
  windowDuration = std::max(windowDuration, minWindowDuration_);

  uint64_t ackBytes = totalAckBytes_;

  if (gotFirstAck_) {
    // Check if we have not received any ACK packet for a while (= for a
    // duration greater than the current window duration). If that is the case
    // then we linearly decrease the bandwidth to zero over a period equal to
    // the current window duration.
    const std::chrono::microseconds timeSinceLastAck =
        duration_cast<std::chrono::microseconds>(Clock::now() -
                                                 ackTimes_[previousIdx]);

    if (timeSinceLastAck > windowDuration) {
      // Linearly decrease the bandwidth to zero over `windowDuration`.
      const float scale = (timeSinceLastAck - windowDuration).count() /
                          static_cast<float>(windowDuration.count());
      const uint64_t bytesToRemove = static_cast<uint64_t>(ackBytes * scale);
      ackBytes = ackBytes > bytesToRemove ? ackBytes - bytesToRemove : 0;
    }
  }

  VLOG(10) << __func__ << ": Computing bandwidth based on " << ackBytes
           << " acknowledged bytes over " << (windowDuration.count() / 1000)
           << " ms";

  return Bandwidth(ackBytes, windowDuration);
}

void RLBandwidthSampler::onPacketAcked(
    const CongestionController::AckEvent &ackEvent, uint64_t rttCounter) {

  if (!gotFirstAck_) {
    // First ACK: we use it to initialize timestamps but ignore acked bytes, as
    // it is difficult to obtain a meaningful bandwidth estimate from one ACK.
    std::fill(ackTimes_.begin(), ackTimes_.end(), ackEvent.ackTime);
    lastEntryInitialTime_ = ackEvent.ackTime;
    gotFirstAck_ = true;
    return;
  }

  // Update number of acked bytes.
  totalAckBytes_ += ackEvent.ackedBytes;

  // We will update rolling window based on how close we are to the previous
  // entry.
  const std::chrono::microseconds lastInterval =
      duration_cast<std::chrono::microseconds>(ackEvent.ackTime -
                                               lastEntryInitialTime_);
  DCHECK(lastInterval.count() >= 0);

  if (lastInterval >= minIntervalBetweenAcks_) {
    // Large enough interval between events: create a new entry.

    // Remove oldest acked bytes.
    totalAckBytes_ -= ackBytes_[ackIdx_];
    // Update initial timestamp for this new entry.
    lastEntryInitialTime_ = ackEvent.ackTime;
    // Store new entry.
    ackBytes_[ackIdx_] = ackEvent.ackedBytes;
    ackTimes_[ackIdx_] = ackEvent.ackTime;
    // Update rolling window index.
    ackIdx_ = (ackIdx_ + 1) % ackBytes_.size();
  } else {
    // New event is too close: combine it with the previous entry.
    const uint64_t previousIdx = getPreviousIdx();
    ackBytes_[previousIdx] += ackEvent.ackedBytes;
    ackTimes_[previousIdx] = ackEvent.ackTime;
  }
}

} // namespace quic