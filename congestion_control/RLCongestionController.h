#pragma once

#include <folly/Optional.h>
#include <quic/QuicException.h>
#include <quic/congestion_control/third_party/windowed_filter.h>
#include <quic/state/StateData.h>

#include <limits>

namespace quic {

using namespace std::chrono_literals;

class RLCongestionController : public CongestionController {
 public:
  explicit RLCongestionController(QuicConnectionStateBase& conn);
  void onRemoveBytesFromInflight(uint64_t) override;
  void onPacketSent(const OutstandingPacket& packet) override;
  void onPacketAckOrLoss(folly::Optional<AckEvent>,
                         folly::Optional<LossEvent>) override;

  uint64_t getWritableBytes() const noexcept override;
  uint64_t getCongestionWindow() const noexcept override;
  CongestionControlType type() const noexcept override;

  uint64_t getBytesInFlight() const noexcept;

  void setConnectionEmulation(uint8_t) noexcept override;
  void setAppIdle(bool, TimePoint) noexcept override;
  void setAppLimited() override;

  bool canBePaced() const noexcept override;

  uint64_t getPacingRate(TimePoint currentTime) noexcept override;

  void markPacerTimeoutScheduled(TimePoint currentTime) noexcept override;

  std::chrono::microseconds getPacingInterval() const noexcept override;

  void setMinimalPacingInterval(std::chrono::microseconds) noexcept override;

  bool isAppLimited() const noexcept override;

 private:
  void onPacketAcked(const AckEvent&);
  void onPacketLoss(const LossEvent&);

  QuicConnectionStateBase& conn_;
  uint64_t bytesInFlight_{0};
  uint64_t cwndBytes_;

  // Copa-style RTT filters to get more accurate min and standing RTT values.
  WindowedFilter<std::chrono::microseconds,
                 MinFilter<std::chrono::microseconds>, uint64_t,
                 uint64_t>
      minRTTFilter_;  // To get min RTT over 10 seconds

  WindowedFilter<std::chrono::microseconds,
                 MinFilter<std::chrono::microseconds>, uint64_t,
                 uint64_t>
      standingRTTFilter_;  // To get min RTT over srtt/2
};

}  // namespace quic
