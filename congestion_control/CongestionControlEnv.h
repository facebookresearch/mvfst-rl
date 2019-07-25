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

  struct Action {
    // This assumes that the policy has a no-op action at index 0
    int32_t cwndAction{0};
  };

  struct Observation {
   public:
    // NOTE: If fields are added, make sure to update fieldsToString() as well.
    // TODO (viswanath): Analyze logs and update fields to get them to somewhat
    // similar scales (packets vs bytes, etc)
    enum class Field : uint16_t {
      // RTT related
      RTT_MIN_MS = 0,
      RTT_STANDING_MS,
      LRTT_MS,
      SRTT_MS,
      RTT_VAR_MS,
      DELAY_MS,

      // Bytes related
      CWND_BYTES,
      BYTES_IN_FLIGHT,
      WRITABLE_BYTES,
      BYTES_SENT,
      BYTES_RECEIVED,
      BYTES_RETRANSMITTED,

      // LossState
      PTO_COUNT,
      TOTAL_PTO_DELTA,  // Derived from LossState::totalPTOCount
      RTX_COUNT,
      TIMEOUT_BASED_RTX_COUNT,

      // AckEvent
      ACKED_BYTES,
      ACKED_PACKETS,
      THROUGHPUT,

      // LossEvent
      LOST_BYTES,
      LOST_PACKETS,
      PERSISTENT_CONGESTION,

      // Previous action taken
      PREV_CWND_ACTION,

      // Total number of fields
      NUM_FIELDS
    };

    static constexpr uint16_t kNumFields =
        static_cast<uint16_t>(Field::NUM_FIELDS);

    Observation() : data_(kNumFields, 0.0) {}

    inline const float* data() const { return data_.data(); }
    inline constexpr uint16_t size() const { return kNumFields; }

    inline float operator[](int idx) const { return data_[idx]; }
    inline float operator[](Field field) const {
      return data_[static_cast<int>(field)];
    }
    inline float& operator[](int idx) { return data_[idx]; }
    inline float& operator[](Field field) {
      return data_[static_cast<int>(field)];
    }

    inline void setField(const Field field, const float& value) {
      data_[static_cast<int>(field)] = value;
    }

    static float reward(const std::vector<Observation>& observations);

    torch::Tensor toTensor() const;
    void toTensor(torch::Tensor& tensor) const;
    static torch::Tensor toTensor(const std::vector<Observation>& observations);
    static void toTensor(const std::vector<Observation>& observations,
                         torch::Tensor& tensor);

    static std::string fieldToString(const uint16_t field);
    static std::string fieldToString(const Field field);

   private:
    std::vector<float> data_;
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
  Action prevAction_;
};

std::ostream& operator<<(std::ostream& os,
                         const CongestionControlEnv::Observation& observation);

}  // namespace quic
