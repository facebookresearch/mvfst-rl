#pragma once

#include <condition_variable>
#include <thread>

#include "rpcenv.grpc.pb.h"
#include "rpcenv.pb.h"

#include "CongestionControlEnv.h"

namespace quic {

class CongestionControlRPCEnv : public CongestionControlEnv {
 public:
  CongestionControlRPCEnv(const CongestionControlEnv::Config& cfg,
                          CongestionControlEnv::Callback* cob,
                          const QuicConnectionStateBase& conn);
  ~CongestionControlRPCEnv() override;

 private:
  // CongestionControlEnv impl
  void onObservation(const std::vector<Observation>& observations) override;

  void loop(const std::string& address);

  static void fillNDArray(rpcenv::NDArray* ndarray,
                          const torch::Tensor& tensor);

  std::unique_ptr<std::thread> thread_;  // Thread to run the gRPC client in
  bool connected_{false};  // Whether we are connected to gRPC server
  std::atomic<bool> shutdown_{false};  // Signals termination of env loop

  torch::Tensor tensor_;  // Tensor holding observations
  float reward_;
  bool observationReady_{false};

  // CV and mutex for co-ordination with gRPC thread.
  std::condition_variable cv_;
  std::mutex mutex_;
};

}  // namespace quic
