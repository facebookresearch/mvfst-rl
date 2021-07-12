/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <glog/logging.h>
#include <quic/QuicConstants.h>
#include <quic/congestion_control/CongestionControllerFactory.h>

#include <memory>

#include "RLCongestionController.h"

namespace quic {

struct CongestionController;
struct QuicConnectionStateBase;

class RLCongestionControllerFactory : public CongestionControllerFactory {
public:
  RLCongestionControllerFactory(
      std::shared_ptr<CongestionControlEnvFactory> envFactory)
      : envFactory_(envFactory) {
    CHECK_NOTNULL(envFactory.get());
  }

  ~RLCongestionControllerFactory() override = default;

  std::unique_ptr<CongestionController>
  makeCongestionController(QuicConnectionStateBase &conn,
                           CongestionControlType type) {
    const char *actorId = std::getenv("MVFSTRL_ACTOR_ID");
    const char *actorPID = std::getenv("MVFSTRL_ACTOR_PID");
    LOG(INFO) << "Creating RLCongestionController (actorId: "
              << (actorId != nullptr ? actorId : "?")
              << ", actorPID: " << (actorPID != nullptr ? actorPID : "?")
              << ")";
    return std::make_unique<RLCongestionController>(conn, envFactory_);
  }

private:
  std::shared_ptr<CongestionControlEnvFactory> envFactory_;
};

} // namespace quic
