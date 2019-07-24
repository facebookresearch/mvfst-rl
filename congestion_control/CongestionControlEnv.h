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

  // Action space
  struct Action {
    // This assumes that the policy has a no-op action at index 0
    int32_t cwndAction{0};
  };

  // Observation space
  //
  // NOTE: If fields are added/removed, remember to also update the following:
  // 1. Observation::DIMS count
  // 2. Serialization implementation in Observation::toTensor().
  // 3. Observation::operator<<() implementation.
  struct Observation {
    // RTT related
    float rttMinMs{0.0};
    float rttStandingMs{0.0};
    float lrttMs{0.0};
    float srttMs{0.0};
    float rttVarMs{0.0};
    float delayMs{0.0};

    // Bytes related
    uint64_t cwndBytes{0};
    uint64_t bytesInFlight{0};
    uint64_t writableBytes{0};
    uint64_t bytesSent{0};
    uint64_t bytesRecvd{0};
    uint64_t bytesRetransmitted{0};

    // LossState
    uint32_t ptoCount{0};
    uint32_t totalPTODelta{0};  // Derived from LossState::totalPTOCount
    uint32_t rtxCount{0};
    uint32_t timeoutBasedRtxCount{0};

    // AckEvent
    uint64_t ackedBytes{0};
    uint32_t ackedPackets{0};
    float throughput{0};

    // LossEvent
    uint64_t lostBytes{0};
    uint32_t lostPackets{0};
    bool persistentCongestion{false};

    // Previous action taken
    struct Action lastAction;

    static const int DIMS = 23;
    // TODO slog print

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
  void onAction(const Action& action);
  void onReset();

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
  Action lastAction_;
};

}  // namespace quic
