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
// This should be polybeast.py --hidden_size + 1
const int kLSTMHiddenSize = 512 + 1;
}

CongestionControlLocalEnv::CongestionControlLocalEnv(
    const Config& cfg, Callback* cob, const QuicConnectionStateBase& conn)
    : CongestionControlEnv(cfg, cob, conn) {
  LOG(INFO) << "Loading traced model from " << cfg.modelFile;
  module_ = torch::jit::load(cfg.modelFile, at::kCPU);

  thread_ =
      std::make_unique<std::thread>(&CongestionControlLocalEnv::loop, this);
}

CongestionControlLocalEnv::~CongestionControlLocalEnv() {
  shutdown_ = true;
  thread_->join();
}

void CongestionControlLocalEnv::onObservation(Observation&& obs, float reward) {
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
      {torch::zeros({1, 1, kLSTMHiddenSize}, at::kFloat),
       torch::zeros({1, 1, kLSTMHiddenSize}, at::kFloat)});

  while (!shutdown_) {
    cv_.wait(lock, [&]() -> bool { return (observationReady_ || shutdown_); });
    if (shutdown_) {
      LOG(INFO) << "Inference loop terminating after " << episode_step
                << " steps, total return = " << episode_return;
      return;
    }

    done = (episode_step == 0);
    episode_return += reward_;
    VLOG(2) << "Episode step = " << episode_step
            << ", total return = " << episode_return;

    auto reward_tensor = torch::from_blob(&reward_, {1, 1}, at::kFloat);
    auto done_tensor = torch::from_blob(&done, {1, 1}, at::kByte);

    c10::Dict<std::string, torch::Tensor> input_dict;
    input_dict.insert("frame", tensor_.reshape({1, 1, -1}));
    input_dict.insert("reward", std::move(reward_tensor));
    input_dict.insert("done", std::move(done_tensor));

    std::vector<torch::IValue> inputs{std::move(input_dict),
                                      std::move(core_state)};
    const auto& outputs = module_.forward(inputs).toTuple();

    // Outputs: ((action, policy_logits, baseline), core_state)
    const auto& action_tensor =
        outputs->elements()[0].toTuple()->elements()[0].toTensor();
    core_state = outputs->elements()[1].toTuple();

    action.cwndAction = action_tensor[0][0].item<long>();
    onAction(action);

    episode_step++;
    observationReady_ = false;  // Back to waiting
  }
}

}  // namespace quic
