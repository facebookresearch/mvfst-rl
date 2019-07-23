#include "CongestionControlRPCEnv.h"

using namespace grpc;
using namespace rpcenv;

namespace quic {

CongestionControlRPCEnv::CongestionControlRPCEnv(
    const CongestionControlEnv::Config& config,
    CongestionControlEnv::Callback* cob)
    : CongestionControlEnv(config, cob),
      envServer_(std::make_unique<EnvServer>(this, config.rpcPort)) {
  tensor_ = torch::empty({0, Observation::DIMS}, torch::kFloat32);
  envServer_->start();
}

CongestionControlRPCEnv::~CongestionControlRPCEnv() {
  shutdown_ = true;
  envServer_->stop();
}

void CongestionControlRPCEnv::onObservation(
    const std::vector<Observation>& observations) {
  float reward = 0;
  for (const auto& observation : observations) {
    reward += observation.reward();
  }

  {
    std::lock_guard<std::mutex> g(mutex_);
    Observation::toTensor(observations, tensor_);
    reward_ = reward;
    observationReady_ = true;
  }
  cv_.notify_one();
}

grpc::Status CongestionControlRPCEnv::StreamingEnv(
    ServerContext* context,
    ServerReaderWriter<rpcenv::Step, rpcenv::Action>* stream) {
  LOG(INFO) << "StreamingEnv initiated";

  rpcenv::Step step_pb;
  rpcenv::Action action_pb;
  Action action;
  int episode_step = 0;
  float episode_return = 0.0;
  std::unique_lock<std::mutex> lock(mutex_);

  while (!shutdown_) {
    step_pb.Clear();

    cv_.wait(lock, [&]() -> bool { return (observationReady_ || shutdown_); });
    if (shutdown_) {
      LOG(INFO) << "StreamingEnv terminating";
      return grpc::Status::OK;
    }

    episode_step += 1;
    episode_return += reward_;
    fillNDArray(step_pb.mutable_observation()->mutable_array(), tensor_);
    step_pb.set_reward(reward_);
    step_pb.set_done(false);  // TODO (viswanath): Are we ever done?
    step_pb.set_episode_step(episode_step);
    step_pb.set_episode_return(episode_return);

    // TODO (viswanath): Think of scenarios where onObservation is too fast
    // and has another state update before stream->Read() gets back.
    observationReady_ = false;  // Back to waiting
    lock.unlock();

    stream->Write(step_pb);
    if (!stream->Read(&action_pb)) {
      throw std::runtime_error("Read failed in StreamingEnv");
    }

    action.cwndAction = action_pb.action();
    onAction(action);
  }

  return grpc::Status::OK;
}

void CongestionControlRPCEnv::fillNDArray(rpcenv::NDArray* ndarray,
                                          const torch::Tensor& tensor) {
  // 11 -> float32. Hacky, but we can't use NPY_* defines as
  // we don't have a Python interpreter here.
  ndarray->set_dtype(11);
  for (const auto& dim : tensor.sizes()) {
    ndarray->add_shape(dim);
  }
  ndarray->set_data(tensor.contiguous().data_ptr(), tensor.nbytes());
}

}  // namespace quic
