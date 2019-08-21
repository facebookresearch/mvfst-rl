#pragma once

#include "CongestionControlRPCEnv.h"
#include "CongestionControlRandomEnv.h"

namespace quic {

class CongestionControlEnvFactory {
 public:
  CongestionControlEnvFactory(const CongestionControlEnv::Config& cfg)
      : cfg_(cfg) {}

  std::unique_ptr<CongestionControlEnv> make(
      CongestionControlEnv::Callback* cob,
      const QuicConnectionStateBase& conn) {
    switch (cfg_.mode) {
      case CongestionControlEnv::Config::Mode::LOCAL:
        LOG(FATAL) << "Local mode not yet implemented";
        return nullptr;
      case CongestionControlEnv::Config::Mode::REMOTE:
        return std::make_unique<CongestionControlRPCEnv>(cfg_, cob, conn);
      case CongestionControlEnv::Config::Mode::RANDOM:
        return std::make_unique<CongestionControlRandomEnv>(cfg_, cob, conn);
      default:
        LOG(FATAL) << "Unknown mode";
        return nullptr;
    }
    __builtin_unreachable();
  }

 private:
  CongestionControlEnv::Config cfg_;
};

}  // namespace quic
