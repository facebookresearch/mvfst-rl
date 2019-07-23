#pragma once

#include <folly/io/async/EventBaseManager.h>
#include <folly/io/async/HHWheelTimer.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <vector>

namespace quic {

class CongestionControlEnv {
 public:
  enum class Mode : uint8_t {
    TRAIN = 0,
    TEST,
  };

  // Type of aggregation to group state updates
  enum class Aggregation : uint8_t {
    TIME_WINDOW = 0,  // Group state updates every X ms
    FIXED_WINDOW,     // Group every Y state updates
    // TODO: Other kinds of aggregation (like avg/max/ewma)?
  };

  struct Config {
    Mode mode{Mode::TRAIN};
    uint16_t rpcPort{60000};  // Port for RPCEnv
    Aggregation aggregation{Aggregation::TIME_WINDOW};
    std::chrono::milliseconds windowDuration{500};  // Time window duration
    uint32_t windowSize{10};                        // Fixed window size
    uint32_t stepsPerEpisode{400};  // Reset interval for env during training
  };

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
    virtual void onReset() noexcept = 0;
  };

  CongestionControlEnv(const Config& config, Callback* cob);
  virtual ~CongestionControlEnv() = default;

  // To be invoked by whoever owns CongestionControlEnv (such as
  // RLCongestionController) to share Observation updates after every
  // Ack/Loss event.
  void onUpdate(Observation&& observation);

 protected:
  // onObservation() will be triggered when there are enough state updates to
  // run the policy and predict an action. Subclasses should implement this
  // and return the action via onAction() callback, either synchronously or
  // asynchronously.
  virtual void onObservation(const std::vector<Observation>& observations) = 0;

  // Callbacks to be invoked by subclasses when there is an update
  // following onObservation().
  void onAction(const Action& action) const;
  void onReset() const;

  const Config& config_;

 private:
  class ObservationTimeout : public folly::HHWheelTimer::Callback {
   public:
    explicit ObservationTimeout(CongestionControlEnv* env)
        : env_(CHECK_NOTNULL(env)),
          evb_(folly::EventBaseManager::get()->getEventBase()) {}
    ~ObservationTimeout() override = default;

    void schedule(const std::chrono::milliseconds& timeoutMs) noexcept {
      evb_->timer().scheduleTimeout(this, timeoutMs);
    }

    void timeoutExpired() noexcept override {
      env_->observationTimeoutExpired();
    }

    void callbackCanceled() noexcept override { return; }

   private:
    CongestionControlEnv* env_;
    folly::EventBase* evb_;
  };

  void observationTimeoutExpired() noexcept;

  Callback* cob_{nullptr};
  std::vector<Observation> observations_;
  ObservationTimeout observationTimeout_;
};

}  // namespace quic
