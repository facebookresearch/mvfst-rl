#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

namespace quic {

class CongestionControlEnv {
 public:
  // Observation space
  struct Observation {
    // TODO (viswanath): Add more stuff
    uint64_t rtt;
    uint64_t cwndBytes;

    static const int DIMS = 2;

    torch::Tensor toTensor() const;
    void toTensor(torch::Tensor& tensor) const;

    static torch::Tensor toTensor(const std::vector<Observation>& observations);
    static void toTensor(const std::vector<Observation>& observations,
                         torch::Tensor& tensor);
  };

  // Action space
  struct Action {
    uint64_t cwndBytes;
  };

  struct Callback {
    virtual ~Callback() = default;
    virtual void onUpdate(const Action& action) noexcept = 0;
  };

  CongestionControlEnv(Callback* cob) : cob_(CHECK_NOTNULL(cob)) {}
  virtual ~CongestionControlEnv() = default;

  static std::unique_ptr<CongestionControlEnv> make(Callback* cob);

  void onObservation(const Observation& observation);

 protected:
  virtual void onReport(const std::vector<Observation>& observations) = 0;

  Callback* cob_{nullptr};
};

}  // namespace quic
