/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include "CongestionControlEnv.h"

namespace quic {

// Basic controller aiming at reaching a specific cwnd target.
// At each decision step, this controller greedily picks the action that brings
// cwnd closest to its target value (NB: it may not be able to exactly reach
// it).
class CongestionControlFixedCwndEnv : public CongestionControlEnv {

public:
  CongestionControlFixedCwndEnv(const Config &cfg, Callback *cob,
                                const QuicConnectionStateBase &conn);

private:
  // Target value for cwnd, in bytes.
  uint64_t cwndBytesTarget_;

  void onObservation(Observation &&obs, float reward) override;

  // Return how far we currently are from the target cwnd value.
  uint64_t distToTarget(uint64_t cwndBytes) const;
};

} // namespace quic
