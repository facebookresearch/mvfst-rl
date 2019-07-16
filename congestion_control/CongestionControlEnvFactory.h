#pragma once

#include "CongestionControlRPCEnv.h"

namespace quic {

class CongestionControlEnvFactory {
 public:
  CongestionControlEnvFactory(const CongestionControlEnv::Config& config)
      : config_(config) {}

  std::unique_ptr<CongestionControlEnv> make(
      CongestionControlEnv::Callback* cob) {
    switch (config_.type) {
      case CongestionControlEnv::Type::RPC:
        return std::make_unique<CongestionControlRPCEnv>(config_, cob);
      case CongestionControlEnv::Type::None:
        std::runtime_error("Unsupported CongestionControlEnv::Type in config");
        break;
    }
  }

 private:
  CongestionControlEnv::Config config_;
};

}  // namespace quic
