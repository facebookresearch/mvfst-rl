#include "CongestionControlRPCEnv.h"

#include <grpc++/grpc++.h>

using namespace grpc;
using namespace rpcenv;

namespace quic {

namespace {
constexpr std::chrono::seconds kConnectTimeout{5};
}

CongestionControlRPCEnv::CongestionControlRPCEnv(
    const Config& cfg, Callback* cob, const QuicConnectionStateBase& conn)
    : CongestionControlEnv(cfg, cob, conn) {
  thread_ = std::make_unique<std::thread>(&CongestionControlRPCEnv::loop, this,
                                          cfg.rpcAddress);
  tensor_ = torch::empty({0}, torch::kFloat32);

  // Wait until connected to gRPC server
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&]() -> bool { return connected_; });
}

CongestionControlRPCEnv::~CongestionControlRPCEnv() {
  shutdown_ = true;
  thread_->join();
}

void CongestionControlRPCEnv::onObservation(Observation&& obs, float reward) {
  {
    std::lock_guard<std::mutex> g(mutex_);
    obs.toTensor(tensor_);
    reward_ = reward;
    observationReady_ = true;
  }
  cv_.notify_one();
}

void CongestionControlRPCEnv::loop(const std::string& address) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  std::unique_ptr<rpcenv::ActorPoolServer::Stub> stub =
      rpcenv::ActorPoolServer::NewStub(channel);

  LOG(INFO) << "Connecting to ActorPoolServer at " << address << " ...";
  const auto& deadline = std::chrono::system_clock::now() + kConnectTimeout;
  if (!channel->WaitForConnected(deadline)) {
    LOG(FATAL) << "Timed out connecting to ActorPoolServer: " << address;
  }

  // Notify that we are connected
  {
    std::lock_guard<std::mutex> g(mutex_);
    connected_ = true;
  }
  cv_.notify_one();
  LOG(INFO) << "Connected to ActorPoolServer: " << address;

  grpc::ClientContext context;
  std::shared_ptr<grpc::ClientReaderWriter<rpcenv::Step, rpcenv::Action>>
      stream(stub->StreamingActor(&context));

  rpcenv::Step step_pb;
  rpcenv::Action action_pb;
  Action action;
  bool done = true;
  uint32_t episode_step = 0;
  float episode_return = 0.0;
  std::unique_lock<std::mutex> lock(mutex_);
  std::chrono::time_point<std::chrono::steady_clock> policyBegin;
  std::chrono::duration<float, std::milli> policyElapsed;

  while (!shutdown_) {
    step_pb.Clear();

    cv_.wait(lock, [&]() -> bool { return (observationReady_ || shutdown_); });
    if (shutdown_) {
      LOG(INFO) << "RPC env loop terminating";
      const auto& status = stream->Finish();
      if (!status.ok()) {
        LOG(ERROR) << "RPC env loop failed on finish.";
      }
      return;
    }

    policyBegin = std::chrono::steady_clock::now();

    // The lifetime of a connection is seen as a single episode, so
    // done is set to true only at the beginning of the episode (to mark
    // the end of the previous episode. Episodic training should be
    // implemented via resetting the entire connection.
    done = (episode_step == 0);
    episode_return += reward_;

    fillNDArray(step_pb.mutable_observation()->mutable_array(), tensor_);
    step_pb.set_reward(reward_);
    step_pb.set_done(done);
    step_pb.set_episode_step(episode_step);
    step_pb.set_episode_return(episode_return);

    episode_step++;

    // In theory, it is possible that onObservation could sometimes be too fast
    // and has another state update before stream->Read() gets back.
    // For now, this would block in onObservation() as the mutex is locked
    // until the next cv_.wait() call. In reality, this shouldn't be a problem
    // as model runtimes are sufficiently fast.
    observationReady_ = false;  // Back to waiting

    stream->Write(step_pb);
    if (!stream->Read(&action_pb)) {
      LOG(FATAL) << "Read failed from gRPC server.";
    }
    action.cwndAction = action_pb.action();
    onAction(action);

    policyElapsed = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - policyBegin);
    VLOG(1) << "Action updated, policy elapsed time = " << policyElapsed.count()
            << " ms";
  }
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
