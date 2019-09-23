/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include "CongestionControlLocalEnv.h"
#include "CongestionControlRandomEnv.h"

#ifndef MVFSTRL_INFERENCE_ONLY
#include "CongestionControlRPCEnv.h"
#endif

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
        return std::make_unique<CongestionControlLocalEnv>(cfg_, cob, conn);
      case CongestionControlEnv::Config::Mode::REMOTE:
#ifdef MVFSTRL_INFERENCE_ONLY
        LOG(FATAL) << "REMOTE mode is not available as this is an inference "
                      "only build.";
        return nullptr;
#else
        return std::make_unique<CongestionControlRPCEnv>(cfg_, cob, conn);
#endif
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
