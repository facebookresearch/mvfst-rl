#pragma once

#include "CongestionControlRPCEnv.h"
#include "CongestionControlRandomEnv.h"

namespace quic {

class CongestionControlEnvFactory {
 public:
  CongestionControlEnvFactory(const CongestionControlEnv::Config& config)
      : config_(config) {}

  std::unique_ptr<CongestionControlEnv> make(
      CongestionControlEnv::Callback* cob,
      const QuicConnectionStateBase& conn) {
    switch (config_.mode) {
      case CongestionControlEnv::Config::Mode::TRAIN:
        return std::make_unique<CongestionControlRPCEnv>(config_, cob, conn);
      case CongestionControlEnv::Config::Mode::TEST:
        LOG(FATAL) << "Test mode not yet implemented";
        break;
      case CongestionControlEnv::Config::Mode::RANDOM:
        return std::make_unique<CongestionControlRandomEnv>(config_, cob, conn);
    }
  }

 private:
  CongestionControlEnv::Config config_;
};

}  // namespace quic
