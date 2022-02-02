/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <quic/common/TimeUtil.h>
#include <quic/congestion_control/CongestionControlFunctions.h>
#include <quic/congestion_control/Copa.h>

#include "NetworkState.h"
#include "RLCongestionController.h"

namespace quic {

using namespace std::chrono;

using Field = NetworkState::Field;

RLCongestionController::RLCongestionController(
    QuicConnectionStateBase &conn,
    std::shared_ptr<CongestionControlEnvFactory> envFactory)
    : conn_(conn),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen),
      env_(envFactory->make(this, conn)), cfg_(env_->config()),
      minRTTFilter_(kMinRTTWindowLength.count(), 0us, 0), // length reset below
      standingRTTFilter_(100000, 0us, 0),                 // 100ms
      bandwidthSampler_(conn, env_->config()) {
  DCHECK(env_);
  minRTTFilter_.SetWindowLength(cfg_.minRTTWindowLength.count());

  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_ << " "
           << conn_;
}

void RLCongestionController::onRemoveBytesFromInflight(uint64_t bytes) {
  subtractAndCheckUnderflow(bytesInFlight_, bytes);
  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_ << " "
           << conn_;
}

void RLCongestionController::onPacketSent(const OutstandingPacket &packet) {
  addAndCheckOverflow(bytesInFlight_, packet.metadata.encodedSize);

  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_
           << " bytesBuffered=" << conn_.flowControlState.sumCurStreamBufferLen
           << " packetNum=" << packet.packet.header.getPacketSequenceNum()
           << " " << conn_;
}

void RLCongestionController::onPacketAckOrLoss(
    folly::Optional<AckEvent> ack, folly::Optional<LossEvent> loss) {
  if (loss) {
    onPacketLoss(*loss);
  }
  if (ack && ack->largestAckedPacket.hasValue()) {
    onPacketAcked(*ack);
  }

  // State update to the env
  NetworkState obs;
  if (setNetworkState(ack, loss, obs)) {
    env_->onNetworkState(std::move(obs));
  }
}

void RLCongestionController::onPacketAcked(const AckEvent &ack) {
  DCHECK(ack.largestAckedPacket.hasValue());
  subtractAndCheckUnderflow(bytesInFlight_, ack.ackedBytes);
  const uint32_t lrttWithNoise = static_cast<uint32_t>(
      conn_.lossState.lrtt.count() * (1. + normal_(rng_) * cfg_.rttNoiseStd));
  minRTTFilter_.Update(
      std::chrono::microseconds(lrttWithNoise),
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());
  standingRTTFilter_.SetWindowLength(conn_.lossState.srtt.count() / 2);
  standingRTTFilter_.Update(
      conn_.lossState.lrtt,
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());

  // The `rttCounter` argument is set to 0 because it is ignored in
  // `RLBandwidthSampler`. If one wanted to use a different bandwidth estimator
  // (e.g. `BbrBandwidthSampler`) then a proper counter should be implemeneted.
  bandwidthSampler_.onPacketAcked(ack, 0);

  VLOG(10) << __func__ << "ack size=" << ack.ackedBytes
           << " num packets acked=" << ack.ackedBytes / conn_.udpSendPacketLen
           << " writable=" << getWritableBytes() << " cwnd=" << cwndBytes_
           << " inflight=" << bytesInFlight_
           << " sRTT=" << conn_.lossState.srtt.count()
           << " lRTT=" << conn_.lossState.lrtt.count()
           << " mRTT=" << conn_.lossState.mrtt.count()
           << " rttvar=" << conn_.lossState.rttvar.count()
           << " packetsBufferred="
           << conn_.flowControlState.sumCurStreamBufferLen
           << " packetsRetransmitted=" << conn_.lossState.rtxCount << " "
           << conn_;
}

void RLCongestionController::onPacketLoss(const LossEvent &loss) {
  VLOG(10) << __func__ << " lostBytes=" << loss.lostBytes
           << " lostPackets=" << loss.lostPackets << " cwnd=" << cwndBytes_
           << " inflight=" << bytesInFlight_ << " " << conn_;
  DCHECK(loss.largestLostPacketNum.hasValue());
  subtractAndCheckUnderflow(bytesInFlight_, loss.lostBytes);
  if (loss.persistentCongestion) {
    VLOG(10) << __func__ << " writable=" << getWritableBytes()
             << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_ << " "
             << conn_;
  }
}

void RLCongestionController::onUpdate(const uint64_t &cwndBytes) noexcept {
  cwndBytes_ = cwndBytes;
}

bool RLCongestionController::setNetworkState(
    const folly::Optional<AckEvent> &ack,
    const folly::Optional<LossEvent> &loss, NetworkState &obs) {
  const auto &state = conn_.lossState;

  const auto &rttMin = minRTTFilter_.GetBest();
  const auto &rttStanding = standingRTTFilter_.GetBest().count();

  // Compute delay (from possibly noisy RTT measurement).
  float delay;
  double lrttScaling = 1.;
  if (cfg_.rttNoiseStd > 0) {
    lrttScaling = std::max(0., 1. + normal_(rng_) * cfg_.rttNoiseStd);
    const float lrttWithNoise = state.lrtt.count() / 1000.f * lrttScaling;
    delay = std::max(0.f, lrttWithNoise - rttMin.count() / 1000.f);
  } else {
    delay = duration_cast<microseconds>(state.lrtt - rttMin).count() / 1000.f;
  }

  if (rttStanding == 0 || delay < 0) {
    LOG(ERROR)
        << "Invalid rttStanding or delay, skipping network state update: "
        << "rttStanding = " << rttStanding << ", delay = " << delay << "ms "
        << conn_;
    return false;
  }

  const float normMs = env_->normMs();
  const float normBytes = env_->normBytes();

  obs[Field::RTT_MIN] = rttMin.count() / 1000.0 / normMs;
  /*
  obs[Field::RTT_STANDING] = rttStanding / 1000.0 / normMs;
  // `lrtt` is the last observed RTT
  // `ldrtt` is the last observed RTT *including* ACK delay
  // `srtt` is a moving average of observed RTTs with coefficient 1/kRttAlpha
  obs[Field::LRTT] = state.lrtt.count() / 1000.0 / normMs;
  obs[Field::LDRTT] = state.ldrtt.count() / 1000.0 / normMs;
  obs[Field::SRTT] = state.srtt.count() / 1000.0 / normMs;
  obs[Field::RTT_VAR] = state.rttvar.count() / 1000.0 / normMs;
  */
  obs[Field::DELAY] = delay / normMs;

  // The ACK delay is smoothed with an exponential moving average.
  avgAckDelayMs_ = (1.f - cfg_.ackDelayAvgCoeff) * avgAckDelayMs_ +
                   cfg_.ackDelayAvgCoeff * state.lastAckDelay.count() / 1000.f;
  obs[Field::ACK_DELAY] = avgAckDelayMs_ / normMs;

  // Normalize CWND field such that max value is 1.
  obs[Field::CWND] =
      cwndBytes_ / static_cast<float>(conn_.transportSettings.maxCwndInMss *
                                      conn_.udpSendPacketLen);

  // We compute a throughput estimate as CWND / total RTT (including ACK delay).
  // Note that this estimate does *not* account for lost bytes.
  // The total RTT is smoothed with the same coefficient as for
  // `avgAckDelayMs_`.
  const float ldrttWithNoise = state.ldrtt.count() / 1000.f * lrttScaling;
  if (avgRTTWithDelayMs_ <= 0.f) {
    avgRTTWithDelayMs_ = ldrttWithNoise;
  } else {
    avgRTTWithDelayMs_ = (1.f - cfg_.ackDelayAvgCoeff) * avgRTTWithDelayMs_ +
                         cfg_.ackDelayAvgCoeff * ldrttWithNoise;
  }
  obs[Field::NOISY_RTT_WITH_DELAY] = avgRTTWithDelayMs_ / normMs;
  obs[Field::THROUGHPUT_FROM_CWND] =
      cwndBytes_ / std::max(1.f, avgRTTWithDelayMs_) * 1000.f / normBytes;

  /*
  obs[Field::IN_FLIGHT] = bytesInFlight_ / normBytes;
  obs[Field::WRITABLE] = getWritableBytes() / normBytes;
  obs[Field::SENT] = (state.totalBytesSent - prevTotalBytesSent_) / normBytes;
  obs[Field::RECEIVED] =
      (state.totalBytesRecvd - prevTotalBytesRecvd_) / normBytes;
  obs[Field::RETRANSMITTED] =
      (state.totalBytesRetransmitted - prevTotalBytesRetransmitted_) /
      normBytes;
  */

  // The throughput is in bytes / s => we normalize it with `normBytes`.
  DCHECK(bandwidthSampler_.getBandwidth().unitType ==
         Bandwidth::UnitType::BYTES);
  obs[Field::THROUGHPUT] =
      bandwidthSampler_.getBandwidth().normalize() / normBytes;

  /*
  // This is the (log) ratio of the throughput measured from ACK events vs. the
  estimated
  // throughput based on CWND.
  obs[Field::THROUGHPUT_LOG_RATIO] =
      log(1e-5 + obs[Field::THROUGHPUT] / obs[Field::THROUGHPUT_FROM_CWND]);
  */

  /*
  obs[Field::PTO_COUNT] = state.ptoCount;
  obs[Field::TOTAL_PTO_DELTA] = state.totalPTOCount - prevTotalPTOCount_;
  obs[Field::RTX_COUNT] = state.rtxCount - prevRtxCount_;
  obs[Field::TIMEOUT_BASED_RTX_COUNT] =
      state.timeoutBasedRtxCount - prevTimeoutBasedRtxCount_;

  if (ack && ack->largestAckedPacket.hasValue()) {
    obs[Field::ACKED] = ack->ackedBytes / normBytes;
  }

  if (loss) {
    obs[Field::LOST] = loss->lostBytes / normBytes;
    // obs[Field::PERSISTENT_CONGESTION] = loss->persistentCongestion;
  }
  */

  // Update prev state values
  prevTotalBytesSent_ = state.totalBytesSent;
  prevTotalBytesRecvd_ = state.totalBytesRecvd;
  prevTotalBytesRetransmitted_ = state.totalBytesRetransmitted;
  prevTotalPTOCount_ = state.totalPTOCount;
  prevRtxCount_ = state.rtxCount;
  prevTimeoutBasedRtxCount_ = state.timeoutBasedRtxCount;

  return true;
}

uint64_t RLCongestionController::getWritableBytes() const noexcept {
  if (bytesInFlight_ > cwndBytes_) {
    return 0;
  } else {
    return cwndBytes_ - bytesInFlight_;
  }
}

uint64_t RLCongestionController::getCongestionWindow() const noexcept {
  return cwndBytes_;
}

CongestionControlType RLCongestionController::type() const noexcept {
  return CongestionControlType::None;
}

uint64_t RLCongestionController::getBytesInFlight() const noexcept {
  return bytesInFlight_;
}

void RLCongestionController::setAppIdle(bool,
                                        TimePoint) noexcept { /* unsupported */
}

void RLCongestionController::setAppLimited() { /* unsupported */
}

bool RLCongestionController::isAppLimited() const noexcept {
  return false; // not supported
}

} // namespace quic
