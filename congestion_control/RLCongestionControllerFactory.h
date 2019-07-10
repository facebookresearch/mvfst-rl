#pragma once

#include <glog/logging.h>
#include <quic/QuicConstants.h>
#include <quic/congestion_control/CongestionControllerFactory.h>

#include <memory>

#include "congestion_control/RLCongestionController.h"

namespace quic {

struct CongestionController;
struct QuicConnectionStateBase;

class RLCongestionControllerFactory : public CongestionControllerFactory {
 public:
  ~RLCongestionControllerFactory() override = default;

  std::unique_ptr<CongestionController> makeCongestionController(
      QuicConnectionStateBase& conn, CongestionControlType type) {
    LOG(INFO) << "Creating RLCongestionController";
    return std::make_unique<RLCongestionController>(conn);
  }
};

}  // namespace quic
