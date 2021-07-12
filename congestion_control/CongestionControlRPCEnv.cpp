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
using namespace torchbeast;

namespace quic {

namespace {
constexpr std::chrono::seconds kConnectTimeout{5};
}

CongestionControlRPCEnv::CongestionControlRPCEnv(
    const Config &cfg, Callback *cob, const QuicConnectionStateBase &conn)
    : CongestionControlEnv(cfg, cob, conn), actorId_(cfg.actorId) {
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

void CongestionControlRPCEnv::onObservation(Observation &&obs, float reward) {
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

void CongestionControlRPCEnv::loop(const std::string &address) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto stub = RPC::NewStub(channel);

  LOG(INFO) << "Connecting to ActorPoolServer at " << address << " ...";
  const auto &deadline = std::chrono::system_clock::now() + kConnectTimeout;
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
  std::shared_ptr<grpc::ClientReaderWriter<CallRequest, CallResponse>> stream(
      stub->Call(&context));

  Action action;
  bool done = true;
  uint32_t episode_step = 0;
  float episode_return = 0.0;
  CallResponse resp;
  std::unique_lock<std::mutex> lock(mutex_);

  while (!shutdown_) {
    cv_.wait(lock, [&]() -> bool { return (observationReady_ || shutdown_); });
    if (shutdown_) {
      LOG(INFO) << "RPC env loop terminating";
      const auto &status = stream->Finish();
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
    VLOG(2) << "Episode step = " << episode_step
            << ", total return = " << episode_return;

    const auto &req = makeCallRequest(actorId_, cfg_.jobCount, tensor_, reward_, done);
    observationReady_ = false; // Back to waiting

    stream->Write(req);
    if (!stream->Read(&resp)) {
      LOG(FATAL) << "Read failed from gRPC server.";
    }
    if (resp.has_error()) {
      LOG(FATAL) << "Error in response from RL server: "
                 << resp.error().message();
    }
    action.cwndAction = getActionFromCallResponse(resp);
    onAction(action);

    episode_step++;
  }
}

CallRequest CongestionControlRPCEnv::makeCallRequest(int64_t actorId,
                                                     int64_t jobCount,
                                                     const torch::Tensor &obs,
                                                     float reward, bool done) {
  // We need the same run Id across episodes per actor to ensure reconnects to
  // RL server at the beginning of each episode fills the rollout buffer
  // correctly.
  int64_t runId = 0;

  TensorNest actorIdNest(torch::from_blob(&actorId, {}, at::kLong));
  TensorNest runIdNest(torch::from_blob(&runId, {}, at::kLong));
  TensorNest jobCountNest(torch::from_blob(&jobCount, {}, at::kLong));
  TensorNest obsNest(obs);
  TensorNest rewardNest(torch::from_blob(&reward, {}, at::kFloat));
  TensorNest doneNest(torch::from_blob(&done, {}, at::kBool));

  // Input is an ArrayNest of (actor_id, run_id, job_count, (observation, reward, done)).
  TensorNest stepNest(
      quic::utils::vector<TensorNest>({obsNest, rewardNest, doneNest}));
  TensorNest inputs(
      quic::utils::vector<TensorNest>({actorIdNest, runIdNest, jobCountNest, stepNest}));

  CallRequest req;
  req.set_function("inference");
  fill_nest_pb(req.mutable_inputs(), std::move(inputs), fillNDArrayPB);
  return req;
}

uint32_t CongestionControlRPCEnv::getActionFromCallResponse(
    torchbeast::CallResponse &resp) {
  TensorNest output = nest_pb_to_nest(resp.mutable_outputs(), arrayPBToNest);

  // Output should be a single tensor containing the action
  CHECK(output.is_leaf());
  const torch::Tensor &actionTensor = output.front();
  CHECK_EQ(actionTensor.numel(), 1);
  return *actionTensor.data_ptr<long>();
}

void CongestionControlRPCEnv::fillNDArrayPB(NDArray *ndarray,
                                            const torch::Tensor &tensor) {
  for (const auto &dim : tensor.sizes()) {
    ndarray->add_shape(dim);
  }
  ndarray->set_dtype(quic::utils::aten_to_numpy_dtype(tensor.scalar_type()));
  ndarray->set_data(tensor.contiguous().data_ptr(), tensor.nbytes());
}

TensorNest
CongestionControlRPCEnv::arrayPBToNest(torchbeast::NDArray *ndarray) {
  quic::utils::vector<int64_t> shape;
  for (int i = 0, length = ndarray->shape_size(); i < length; ++i) {
    shape.push_back(ndarray->shape(i));
  }

  std::string *data = ndarray->release_data();
  at::ScalarType dtype = quic::utils::numpy_dtype_to_aten(ndarray->dtype());

  return TensorNest(torch::from_blob(
      data->data(), shape, [data](void *ptr) { delete data; }, dtype));
}

} // namespace quic
