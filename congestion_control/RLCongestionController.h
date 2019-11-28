/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <folly/Optional.h>
#include <quic/QuicException.h>
#include <quic/congestion_control/third_party/windowed_filter.h>
#include <quic/state/StateData.h>

#include <limits>

#include "CongestionControlEnvFactory.h"

namespace quic {

using namespace std::chrono_literals;

class RLCongestionController : public CongestionController,
                               public CongestionControlEnv::Callback {
 public:
  RLCongestionController(
      QuicConnectionStateBase& conn,
      std::shared_ptr<CongestionControlEnvFactory> envFactory);

  void onRemoveBytesFromInflight(uint64_t) override;
  void onPacketSent(const OutstandingPacket& packet) override;
  void onPacketAckOrLoss(folly::Optional<AckEvent>,
                         folly::Optional<LossEvent>) override;

  uint64_t getWritableBytes() const noexcept override;
  uint64_t getCongestionWindow() const noexcept override;
  CongestionControlType type() const noexcept override;

  uint64_t getBytesInFlight() const noexcept;

  void setAppIdle(bool, TimePoint) noexcept override;
  void setAppLimited() override;

  bool isAppLimited() const noexcept override;

 private:
  void onPacketAcked(const AckEvent&);
  void onPacketLoss(const LossEvent&);

  // CongestionControlEnv::Callback
  void onUpdate(const uint64_t& cwndBytes) noexcept override;

  bool setNetworkState(const folly::Optional<AckEvent>& ack,
                       const folly::Optional<LossEvent>& loss,
                       NetworkState& obs);

  QuicConnectionStateBase& conn_;
  uint64_t bytesInFlight_{0};
  uint64_t cwndBytes_;

  std::unique_ptr<CongestionControlEnv> env_;

  // Copa-style RTT filters to get more accurate min and standing RTT values.
  WindowedFilter<std::chrono::microseconds,
                 MinFilter<std::chrono::microseconds>, uint64_t,
                 uint64_t>
      minRTTFilter_;  // To get min RTT over 10 seconds

  WindowedFilter<std::chrono::microseconds,
                 MinFilter<std::chrono::microseconds>, uint64_t,
                 uint64_t>
      standingRTTFilter_;  // To get min RTT over srtt/2

  // Variables to track conn_.lossState values from previous ack or loss
  // to compute state deltas for current ack or loss
  uint64_t prevTotalBytesSent_{0};
  uint64_t prevTotalBytesRecvd_{0};
  uint64_t prevTotalBytesRetransmitted_{0};
  uint32_t prevTotalPTOCount_{0};
  uint32_t prevRtxCount_{0};
  uint32_t prevTimeoutBasedRtxCount_{0};
};

}  // namespace quic
