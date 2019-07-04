#include "CongestionControlEnv.h"
#include "CongestionControlRPCEnv.h"

namespace quic {

std::unique_ptr<CongestionControlEnv> CongestionControlEnv::make(
    CongestionControlEnv::Callback* cob) {
  // TODO (viswanath): Add config
  return std::make_unique<CongestionControlRPCEnv>(cob);
}

void CongestionControlEnv::onReport(const Observation& observation) {
  // TODO (viswanath): impl
}

}  // namespace quic
