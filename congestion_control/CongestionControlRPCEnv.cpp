#include "CongestionControlRPCEnv.h"

using namespace grpc;
using namespace rpcenv;

namespace quic {

CongestionControlRPCEnv::CongestionControlRPCEnv(
    CongestionControlEnv::Callback* cob, int port)
    : CongestionControlEnv(cob),
      envServer_(std::make_unique<EnvServer>(this, port)) {
  tensor_ = torch::empty({0, Observation::DIMS}, torch::kFloat32);
  envServer_->start();
}

CongestionControlRPCEnv::~CongestionControlRPCEnv() { envServer_->stop(); }

void CongestionControlRPCEnv::onReport(
    const std::vector<Observation>& observations) {
  Observation::toTensor(observations, tensor_);
  // TODO (viswanath): impl
}

grpc::Status CongestionControlRPCEnv::StreamingEnv(
    ServerContext* context,
    ServerReaderWriter<rpcenv::Step, rpcenv::Action>* stream) {
  LOG(INFO) << "StreamingEnv initiated";

  // TODO (viswanath): impl
  return grpc::Status::OK;
}

}  // namespace quic
