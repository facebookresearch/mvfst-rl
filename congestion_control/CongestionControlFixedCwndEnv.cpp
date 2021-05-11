/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include "CongestionControlFixedCwndEnv.h"

namespace quic {

CongestionControlFixedCwndEnv::CongestionControlFixedCwndEnv(
    const Config &cfg, Callback *cob, const QuicConnectionStateBase &conn)
    : CongestionControlEnv(cfg, cob, conn) {
  cwndBytesTarget_ = cfg_.fixedCwnd * conn.udpSendPacketLen;
}

void CongestionControlFixedCwndEnv::onObservation(Observation &&obs,
                                                  float reward) {
  // Obtain current cwnd value from observation. Compared to directly accessing
  // the `cwndBytes_` attribute, this is closer to how a "real" RL-based policy
  // would work, and avoids potential thread safety issues since `cwndBytes_` is
  // updated asynchronously.
  DCHECK(!obs.history.empty());
  const float lastCwndObs = obs.history.back().cwnd;
  // Convert value to bytes (rounded).
  const uint64_t currentCwndBytes =
      static_cast<uint64_t>(lastCwndObs * normBytes() + 0.5f);

  Action action; // the action we will take

  // How far are we from the target cwnd value?
  uint64_t currentDist = distToTarget(currentCwndBytes);
  if (currentDist == 0) {
    action.cwndAction = 0; // already at target => do nothing
    onAction(action);
    return;
  }

  // Find the action that brings `cwnd` closest to the desired target.
  uint32_t bestActionIdx = 0; // default = do nothing
  for (uint32_t actionIdx = 1; actionIdx < cfg_.actions.size(); ++actionIdx) {
    const uint64_t newCwndBytes =
        getUpdatedCwndBytes(currentCwndBytes, actionIdx);
    const uint64_t newDist = distToTarget(newCwndBytes);
    if (newDist < currentDist) {
      currentDist = newDist;
      bestActionIdx = actionIdx;
    }
  }

  // Apply the selected action.
  action.cwndAction = bestActionIdx;
  onAction(action);
}

uint64_t CongestionControlFixedCwndEnv::distToTarget(uint64_t cwndBytes) const {
  // Compute abs(value - target) safely (they are unsigned integers).
  return cwndBytes > cwndBytesTarget_ ? cwndBytes - cwndBytesTarget_
                                      : cwndBytesTarget_ - cwndBytes;
}

} // namespace quic
