#pragma once

#include "CongestionControlEnv.h"

#include <cstdlib>

namespace quic {

class CongestionControlRandomEnv : public CongestionControlEnv {
 public:
  CongestionControlRandomEnv(const CongestionControlEnv::Config& config,
                             CongestionControlEnv::Callback* cob,
                             const QuicConnectionStateBase& conn)
      : CongestionControlEnv(config, cob, conn) {}

 private:
  // CongestionControlEnv impl
  void onObservation(const std::vector<Observation>& observations) override {
    // Random action
    Action action;
    action.cwndAction = std::rand() % config_.actions.size();
    onAction(std::move(action));
  }
};

}  // namespace quic
