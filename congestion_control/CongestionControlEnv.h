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

    inline float reward() const {
      // TODO (viswanath): impl copa?
      return 0;
    }

    torch::Tensor toTensor() const;
    void toTensor(torch::Tensor& tensor) const;

    static torch::Tensor toTensor(const std::vector<Observation>& observations);
    static void toTensor(const std::vector<Observation>& observations,
                         torch::Tensor& tensor);
  };

  // Action space
  struct Action {
    int32_t cwndAction;
  };

  struct Callback {
    virtual ~Callback() = default;
    virtual void onUpdate(const uint64_t& cwndBytes) noexcept = 0;
  };

  CongestionControlEnv(Callback* cob) : cob_(CHECK_NOTNULL(cob)) {}
  virtual ~CongestionControlEnv() = default;

  static std::unique_ptr<CongestionControlEnv> make(Callback* cob);

  // To be invoked by whoever owns CongestionControlEnv (such as
  // RLCongestionController) to share Observation updates after every
  // Ack/Loss event.
  void onUpdate(const Observation& observation);

 protected:
  // onObservation() will be triggered when there are enough state updates to
  // run the policy and predict an action. Subclasses should implement this
  // and return the action via onAction() callback, either synchronously or
  // asynchronously.
  virtual void onObservation(const std::vector<Observation>& observations) = 0;

  // Callback to be invoked by subclasses when there is an action update
  // following onObservation().
  void onAction(const Action& action);

 private:
  Callback* cob_{nullptr};
};

}  // namespace quic
