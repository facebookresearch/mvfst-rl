#pragma once

#include <glog/logging.h>
#include <memory>

namespace quic {

class CongestionControlEnv {
 public:
  // Observation space
  struct Observation {
    // TODO (viswanath): Add more stuff
    uint64_t rtt;
    uint64_t cwndBytes;
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

  // TODO (viswanath): Add timeout / window aggregation
  void onReport(const Observation& observation);

 protected:
  Callback* cob_{nullptr};
};

}  // namespace quic
