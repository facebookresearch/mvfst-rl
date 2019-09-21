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

#include <cstdlib>

namespace quic {

class CongestionControlRandomEnv : public CongestionControlEnv {
 public:
  CongestionControlRandomEnv(const Config& cfg, Callback* cob,
                             const QuicConnectionStateBase& conn)
      : CongestionControlEnv(cfg, cob, conn) {}

 private:
  // CongestionControlEnv impl
  void onObservation(Observation&& obs, float reward) override {
    // Random action
    Action action;
    action.cwndAction = std::rand() % cfg_.actions.size();
    onAction(action);
  }
};

}  // namespace quic
