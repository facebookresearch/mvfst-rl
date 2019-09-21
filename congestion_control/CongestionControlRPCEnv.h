/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <condition_variable>
#include <thread>

#include "rpcenv.grpc.pb.h"
#include "rpcenv.pb.h"

#include "CongestionControlEnv.h"

namespace quic {

class CongestionControlRPCEnv : public CongestionControlEnv {
 public:
  CongestionControlRPCEnv(const Config& cfg, Callback* cob,
                          const QuicConnectionStateBase& conn);
  ~CongestionControlRPCEnv() override;

 private:
  // CongestionControlEnv impl
  void onObservation(Observation&& obs, float reward) override;

  void loop(const std::string& address);

  static void fillNDArray(rpcenv::NDArray* ndarray,
                          const torch::Tensor& tensor);

  std::unique_ptr<std::thread> thread_;  // Thread to run the gRPC client in
  bool connected_{false};  // Whether we are connected to gRPC server
  std::atomic<bool> shutdown_{false};  // Signals termination of env loop

  // Tensor for holding observations
  torch::Tensor tensor_{torch::empty({0}, torch::kFloat32)};
  float reward_;
  bool observationReady_{false};

  // CV and mutex for co-ordination with gRPC thread.
  std::condition_variable cv_;
  std::mutex mutex_;
};

}  // namespace quic
