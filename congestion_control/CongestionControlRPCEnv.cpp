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

CongestionControlRPCEnv::~CongestionControlRPCEnv() {
  shutdown_ = true;
  envServer_->stop();
}

void CongestionControlRPCEnv::onObservation(
    const std::vector<Observation>& observations) {
  {
    std::lock_guard<std::mutex> g(mutex_);
    Observation::toTensor(observations, tensor_);
    observationReady_ = true;
  }
  cv_.notify_one();
}

grpc::Status CongestionControlRPCEnv::StreamingEnv(
    ServerContext* context,
    ServerReaderWriter<rpcenv::Step, rpcenv::Action>* stream) {
  LOG(INFO) << "StreamingEnv initiated";

  rpcenv::Action action_pb;
  Action action;
  std::unique_lock<std::mutex> lock(mutex_);

  while (!shutdown_) {
    cv_.wait(lock, [&]() -> bool { return (observationReady_ || shutdown_); });
    if (shutdown_) {
      return grpc::Status::OK;
    }
    // TODO (viswanath): stream write
    observationReady_ = false;  // Back to waiting
    lock.unlock();
    // TODO (viswanath): Think of scenarios where onObservation is too fast
    // and has another state update before stream->Read() gets back.

    if (!stream->Read(&action_pb)) {
      throw std::runtime_error("Read failed in StreamingEnv");
    }

    action.cwndAction = action_pb.action();
    onAction(action);
  }

  return grpc::Status::OK;
}

}  // namespace quic
