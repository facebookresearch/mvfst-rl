/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <torch/script.h>
#include <condition_variable>
#include <thread>

#include "CongestionControlEnv.h"

namespace quic {

class CongestionControlLocalEnv : public CongestionControlEnv {
 public:
  CongestionControlLocalEnv(const Config& cfg, Callback* cob,
                            const QuicConnectionStateBase& conn);
  ~CongestionControlLocalEnv() override;

 private:
  // CongestionControlEnv impl
  void onObservation(Observation&& obs, float reward) override;

  void loop();

  std::unique_ptr<std::thread> thread_;  // Thread for inference
  std::atomic<bool> shutdown_{false};    // Signals termination of env loop

  // Tensor for holding observations
  torch::Tensor tensor_{torch::empty({0}, torch::kFloat32)};
  float reward_;
  bool observationReady_{false};

  torch::jit::script::Module module_;

  // CV and mutex for co-ordination with the inference thread.
  std::condition_variable cv_;
  std::mutex mutex_;
};

}  // namespace quic
