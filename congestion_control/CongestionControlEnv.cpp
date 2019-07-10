#include "CongestionControlEnv.h"
#include "CongestionControlRPCEnv.h"

#include <torch/torch.h>

namespace quic {

std::unique_ptr<CongestionControlEnv> CongestionControlEnv::make(
    CongestionControlEnv::Callback* cob) {
  // TODO (viswanath): Add config
  return std::make_unique<CongestionControlRPCEnv>(cob);
}

void CongestionControlEnv::onObservation(const Observation& observation) {
  // TODO (viswanath): Add timeout / window aggregation
  onReport({observation});
}

torch::Tensor CongestionControlEnv::Observation::toTensor() const {
  torch::Tensor tensor;
  toTensor(tensor);
  return tensor;
}

void CongestionControlEnv::Observation::toTensor(torch::Tensor& tensor) const {
  toTensor({*this}, tensor);
}

torch::Tensor CongestionControlEnv::Observation::toTensor(
    const std::vector<Observation>& observations) {
  torch::Tensor tensor;
  toTensor(observations, tensor);
  return tensor;
}

void CongestionControlEnv::Observation::toTensor(
    const std::vector<Observation>& observations, torch::Tensor& tensor) {
  tensor.resize_({observations.size(), Observation::DIMS});
  auto tensor_a = tensor.accessor<float, 2>();
  for (int i = 0; i < tensor_a.size(0); ++i) {
    const auto& obs = observations[i];
    tensor_a[i][0] = obs.rtt;
    tensor_a[i][1] = obs.cwndBytes;
    // TODO (viswanath): Add more stuff
  }
}

}  // namespace quic
