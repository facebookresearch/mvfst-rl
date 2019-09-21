/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
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
#include "NetworkState.h"

namespace quic {

class CongestionControlEnv {
 public:
  using Config = CongestionControlEnvConfig;

  struct Callback {
    virtual ~Callback() = default;
    virtual void onUpdate(const uint64_t& cwndBytes) noexcept = 0;
  };

  struct Action {
    // This assumes that the policy has a no-op action at index 0
    uint32_t cwndAction{0};
  };

  struct History {
    Action action;  // Past action taken
    float cwnd;     // Normalized cwnd after applying the action

    History(const Action& a, const float c) : action(a), cwnd(c) {}
  };

  struct Observation {
   public:
    Observation(const Config& cfg) : cfg_(cfg) {}

    torch::Tensor toTensor() const;
    void toTensor(torch::Tensor& tensor) const;

    std::vector<NetworkState> states;
    std::vector<History> history;

   private:
    const Config& cfg_;
  };

  CongestionControlEnv(const Config& cfg, Callback* cob,
                       const QuicConnectionStateBase& conn);
  virtual ~CongestionControlEnv() = default;

  /**
   * To be invoked by whoever owns CongestionControlEnv (such as
   * RLCongestionController) to share network state updates after every
   * Ack/Loss event.
   */
  void onNetworkState(NetworkState&& state);

  inline const Config& config() const { return cfg_; }
  inline float normMs() const { return cfg_.normMs; }
  inline float normBytes() const { return cfg_.normBytes; }

 protected:
  /**
   * onObservation() will be triggered when there are enough state updates to
   * run the policy and predict an action. Subclasses should implement this
   * and return the action via onAction() callback, either synchronously or
   * asynchronously.
   */
  virtual void onObservation(Observation&& obs, float reward) = 0;

  /**
   * Callback to be invoked by subclasses when there is an update
   * following onObservation().
   */
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
  void handleStates();
  float computeReward(const std::vector<NetworkState>& states) const;
  void updateCwnd(const uint32_t actionIdx);

  inline bool useStateSummary() const {
    return cfg_.useStateSummary ||
           (cfg_.aggregation == Config::Aggregation::TIME_WINDOW);
  }

  /**
   * Compute sum, mean, std, min, max for each field.
   */
  std::vector<NetworkState> stateSummary(
      const std::vector<NetworkState>& states);

  Callback* cob_{nullptr};
  const QuicConnectionStateBase& conn_;
  folly::EventBase* evb_{nullptr};
  ObservationTimeout observationTimeout_;

  uint64_t cwndBytes_;
  std::vector<NetworkState> states_;
  std::deque<History> history_;

  // Intermediate tensor to compute state summary
  torch::Tensor summaryTensor_{torch::empty({0}, torch::kFloat32)};

  std::chrono::time_point<std::chrono::steady_clock> lastObservationTime_;
};

std::ostream& operator<<(std::ostream& os,
                         const CongestionControlEnv::Observation& observation);
std::ostream& operator<<(std::ostream& os,
                         const CongestionControlEnv::History& history);

}  // namespace quic
