#pragma once

#include "CongestionControlEnv.h"

#include <cstdlib>

namespace quic {

class CongestionControlRandomEnv : public CongestionControlEnv {
 public:
  CongestionControlRandomEnv(const CongestionControlEnv::Config& cfg,
                             CongestionControlEnv::Callback* cob,
                             const QuicConnectionStateBase& conn)
      : CongestionControlEnv(cfg, cob, conn) {}

 private:
  // CongestionControlEnv impl
  void onObservation(const std::vector<Observation>& observations) override {
    // Random action
    Action action;
    action.cwndAction = std::rand() % cfg_.actions.size();
    onAction(std::move(action));
  }
};

}  // namespace quic
