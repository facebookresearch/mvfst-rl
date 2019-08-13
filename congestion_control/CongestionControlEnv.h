#pragma once

#include <folly/io/async/EventBaseManager.h>
#include <folly/io/async/HHWheelTimer.h>
#include <glog/logging.h>
#include <quic/state/StateData.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <vector>

#include "CongestionControlEnvConfig.h"

namespace quic {

class CongestionControlEnv {
 public:
  using Config = CongestionControlEnvConfig;

  struct Action {
    // This assumes that the policy has a no-op action at index 0
    uint32_t cwndAction{0};
  };

  struct Observation {
   public:
    // NOTE: If fields are added, make sure to update fieldsToString() as well.
    enum class Field : uint16_t {
      // RTT related
      RTT_MIN = 0,
      RTT_STANDING,
      LRTT,
      SRTT,
      RTT_VAR,
      DELAY,

      // Bytes related
      CWND,
      IN_FLIGHT,
      WRITABLE,
      SENT,
      RECEIVED,
      RETRANSMITTED,

      // LossState
      PTO_COUNT,
      TOTAL_PTO_DELTA,  // Derived from LossState::totalPTOCount
      RTX_COUNT,
      TIMEOUT_BASED_RTX_COUNT,

      // AckEvent
      ACKED,
      THROUGHPUT,

      // LossEvent
      LOST,
      PERSISTENT_CONGESTION,

      // Total number of fields
      NUM_FIELDS
    };

    Observation(const Config& cfg)
        : cfg_(cfg),
          size_(static_cast<size_t>(Field::NUM_FIELDS) +
                cfg.numPastActions * cfg.actions.size()),
          data_(size_, 0.0) {}

    inline const float* data() const { return data_.data(); }
    inline constexpr size_t size() const { return size_; }

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

    template <class Container>
    void setPastActions(const Container& pastActions) {
      CHECK_EQ(pastActions.size(), cfg_.numPastActions);

      // Encode past actions as one-hot
      size_t offset = static_cast<size_t>(Field::NUM_FIELDS);
      std::fill(data_.begin() + offset, data_.end(), 0.0);
      for (size_t i = 0; i < cfg_.numPastActions; ++i) {
        data_[offset + pastActions[i].cwndAction] = 1.0;
        offset += cfg_.actions.size();
      }
    }

    static float reward(const std::vector<Observation>& observations,
                        const Config& cfg);

    torch::Tensor toTensor() const;
    void toTensor(torch::Tensor& tensor) const;
    static torch::Tensor toTensor(const std::vector<Observation>& observations);
    static void toTensor(const std::vector<Observation>& observations,
                         torch::Tensor& tensor);

    static std::string fieldToString(const uint16_t field);
    static std::string fieldToString(const Field field);

   private:
    const Config& cfg_;
    const size_t size_;
    std::vector<float> data_;
  };

  struct Callback {
    virtual ~Callback() = default;
    virtual void onUpdate(const uint64_t& cwndBytes) noexcept = 0;
  };

  CongestionControlEnv(const Config& cfg, Callback* cob,
                       const QuicConnectionStateBase& conn);
  virtual ~CongestionControlEnv() = default;

  inline Observation newObservation() const { return Observation(cfg_); }

  // To be invoked by whoever owns CongestionControlEnv (such as
  // RLCongestionController) to share Observation updates after every
  // Ack/Loss event.
  void onUpdate(Observation&& observation);

  inline const Config& config() const { return cfg_; }

  inline float normMs() const { return cfg_.normMs; }
  inline float normBytes() const { return cfg_.normBytes; }

 protected:
  // onObservation() will be triggered when there are enough state updates to
  // run the policy and predict an action. Subclasses should implement this
  // and return the action via onAction() callback, either synchronously or
  // asynchronously.
  virtual void onObservation(const std::vector<Observation>& observations) = 0;

  // Callback to be invoked by subclasses when there is an update
  // following onObservation().
  void onAction(const Action& action);

  const Config& cfg_;

 private:
  class ObservationTimeout : public folly::HHWheelTimer::Callback {
   public:
    explicit ObservationTimeout(CongestionControlEnv* env,
                                folly::EventBase* evb)
        : env_(CHECK_NOTNULL(env)), evb_(CHECK_NOTNULL(evb)) {}
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

  void updateCwnd(const uint32_t actionIdx);

  Callback* cob_{nullptr};
  const QuicConnectionStateBase& conn_;
  uint64_t cwndBytes_;
  std::vector<Observation> observations_;
  std::deque<Action> pastActions_;
  folly::EventBase* evb_{nullptr};
  ObservationTimeout observationTimeout_;
};

std::ostream& operator<<(std::ostream& os,
                         const CongestionControlEnv::Observation& observation);

}  // namespace quic
