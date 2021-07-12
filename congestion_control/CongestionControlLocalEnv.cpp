/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include "CongestionControlLocalEnv.h"

namespace quic {

namespace {
// This should be flags.hidden_size (+ 1 if flags.use_reward is True)
const int kLSTMHiddenSize = 256;
}

CongestionControlLocalEnv::CongestionControlLocalEnv(
    const Config &cfg, Callback *cob, const QuicConnectionStateBase &conn)
    : CongestionControlEnv(cfg, cob, conn) {
  LOG(INFO) << "Loading traced model from " << cfg.modelFile;
  module_ = torch::jit::load(cfg.modelFile, at::kCPU);

  thread_ =
      std::make_unique<std::thread>(&CongestionControlLocalEnv::loop, this);
}

CongestionControlLocalEnv::~CongestionControlLocalEnv() {
  shutdown_ = true;
  cv_.notify_all();
  thread_->join();
}

void CongestionControlLocalEnv::onObservation(Observation &&obs, float reward) {
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock) {
    LOG(WARNING) << __func__ << ": Still waiting for an update from model, "
                                "skipping observation";
    return;
  }
  obs.toTensor(tensor_);
  reward_ = reward;
  observationReady_ = true;
  lock.unlock();
  cv_.notify_one();
}

void CongestionControlLocalEnv::loop() {
  Action action;
  bool done = true;
  uint32_t episode_step = 0;
  float episode_return = 0.0;
  std::unique_lock<std::mutex> lock(mutex_);

  // Initialize LSTM core state with zeros
  auto core_state = at::ivalue::Tuple::create(
      {torch::zeros({1, kLSTMHiddenSize}, at::kFloat),
       torch::zeros({1, kLSTMHiddenSize}, at::kFloat)});

  while (!shutdown_) {
    cv_.wait(lock, [&]() -> bool { return (observationReady_ || shutdown_); });
    if (shutdown_) {
      break;
    }

    done = (episode_step == 0);
    episode_return += reward_;
    VLOG(2) << "Episode step = " << episode_step
            << ", total return = " << episode_return;

    // env_outputs: (obs, reward, done)
    auto reward_tensor = torch::from_blob(&reward_, {1}, at::kFloat);
    auto done_tensor = torch::from_blob(&done, {1}, at::kBool);
    auto env_outputs = at::ivalue::Tuple::create({tensor_.reshape({1, -1}),
                                                  std::move(reward_tensor),
                                                  std::move(done_tensor)});

    // task observation: although this is not supported in the C++ model, it
    // still needs to be provided as input
    auto task_obs = torch::zeros({1, 0}, at::kFloat);

    // inputs: (last_action, (obs, reward, done), task_obs, core_state)
    auto last_action_tensor =
        torch::from_blob(&action.cwndAction, {1}, at::kLong);
    quic::utils::vector<torch::IValue> inputs{std::move(last_action_tensor),
                                              std::move(env_outputs),
                                              std::move(task_obs),
                                              std::move(core_state)};
    const auto &outputs = module_.forward(inputs).toTuple();

    // output: (action, core_state)
    const auto &action_tensor = outputs->elements()[0].toTensor();
    core_state = outputs->elements()[1].toTuple();

    action.cwndAction = *action_tensor.data_ptr<long>();

    // If there is an ongoing shutdown, it is important not to trigger the action
    // because `onAction()` calls `runImmediatelyOrRunInEventBaseThreadAndWait()`
    // and this method will hang forever during shutdown, preventing the thread from
    // exiting cleanly.
    if (!shutdown_) {
      onAction(action);
    } else {
      LOG(INFO) << "Skipping action due to shutdown in progress";
    }

    episode_step++;
    observationReady_ = false; // Back to waiting
  }

  LOG(INFO) << "Inference loop terminating after " << episode_step
            << " steps, total return = " << episode_return;
}

} // namespace quic
