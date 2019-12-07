/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include "CongestionControlRPCEnv.h"

#include <grpc++/grpc++.h>

#include "Utils.h"

using namespace grpc;
using namespace rpcenv;

namespace quic {

namespace {
constexpr std::chrono::seconds kConnectTimeout{5};
}

CongestionControlRPCEnv::CongestionControlRPCEnv(
    const Config& cfg, Callback* cob, const QuicConnectionStateBase& conn)
    : CongestionControlEnv(cfg, cob, conn) {
  tensor_ = torch::empty({0}, torch::kFloat32);
  thread_ = std::make_unique<std::thread>(&CongestionControlRPCEnv::loop, this,
                                          cfg.rpcAddress);

  // Wait until connected to gRPC server
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&]() -> bool { return connected_; });
}

CongestionControlRPCEnv::~CongestionControlRPCEnv() {
  shutdown_ = true;
  thread_->join();
}

void CongestionControlRPCEnv::onObservation(Observation&& obs, float reward) {
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock) {
    // If we can't acquire the mutex, then we haven't received the action
    // back for the previous observation. Although this should almost never
    // happen as model runtimes are sufficiently fast, we handle this safely
    // here by skipping this observation.
    LOG(WARNING) << __func__ << ": Still waiting for an update from "
                                "ActorPoolServer, skipping observation.";
    return;
  }
  obs.toTensor(tensor_);
  reward_ = reward;
  observationReady_ = true;
  lock.unlock();
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

    observationReady_ = false;  // Back to waiting

    stream->Write(step_pb);
    if (!stream->Read(&action_pb)) {
      LOG(FATAL) << "Read failed from gRPC server.";
    }
    action.cwndAction = action_pb.action();
    onAction(action);
  }
}

void CongestionControlRPCEnv::fillNDArray(rpcenv::NDArray* ndarray,
                                          const torch::Tensor& tensor) {
  for (const auto& dim : tensor.sizes()) {
    ndarray->add_shape(dim);
  }
  ndarray->set_dtype(quic::utils::aten_to_numpy_dtype(tensor.scalar_type()));
  ndarray->set_data(tensor.contiguous().data_ptr(), tensor.nbytes());
}

}  // namespace quic
