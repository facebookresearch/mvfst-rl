#include <quic/common/TimeUtil.h>
#include <quic/congestion_control/CongestionControlFunctions.h>
#include <quic/congestion_control/Copa.h>

#include <congestion_control/RLCongestionController.h>

namespace quic {

using namespace std::chrono;

using Observation = CongestionControlEnv::Observation;
using Field = CongestionControlEnv::Observation::Field;

RLCongestionController::RLCongestionController(
    QuicConnectionStateBase& conn,
    std::shared_ptr<CongestionControlEnvFactory> envFactory)
    : conn_(conn),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen),
      env_(envFactory->make(this)),
      minRTTFilter_(kMinRTTWindowLength.count(), 0us, 0),
      standingRTTFilter_(100000, /*100ms*/
                         0us, 0) {
  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_.load() << " inflight=" << bytesInFlight_
           << " " << conn_;
}

void RLCongestionController::onRemoveBytesFromInflight(uint64_t bytes) {
  subtractAndCheckUnderflow(bytesInFlight_, bytes);
  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_.load() << " inflight=" << bytesInFlight_
           << " " << conn_;
}

void RLCongestionController::onPacketSent(const OutstandingPacket& packet) {
  addAndCheckOverflow(bytesInFlight_, packet.encodedSize);

  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_.load() << " inflight=" << bytesInFlight_
           << " bytesBufferred=" << conn_.flowControlState.sumCurStreamBufferLen
           << " packetNum="
           << folly::variant_match(
                  packet.packet.header,
                  [](auto& h) { return h.getPacketSequenceNum(); })
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
  Observation observation;
  setObservation(ack, loss, observation);
  env_->onUpdate(std::move(observation));
}

void RLCongestionController::onPacketAcked(const AckEvent& ack) {
  DCHECK(ack.largestAckedPacket.hasValue());
  subtractAndCheckUnderflow(bytesInFlight_, ack.ackedBytes);
  minRTTFilter_.Update(
      conn_.lossState.lrtt,
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());
  standingRTTFilter_.SetWindowLength(conn_.lossState.srtt.count() / 2);
  standingRTTFilter_.Update(
      conn_.lossState.lrtt,
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());

  VLOG(10) << __func__ << "ack size=" << ack.ackedBytes
           << " num packets acked=" << ack.ackedBytes / conn_.udpSendPacketLen
           << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_.load() << " inflight=" << bytesInFlight_
           << " sRTT=" << conn_.lossState.srtt.count()
           << " lRTT=" << conn_.lossState.lrtt.count()
           << " mRTT=" << conn_.lossState.mrtt.count()
           << " rttvar=" << conn_.lossState.rttvar.count()
           << " packetsBufferred="
           << conn_.flowControlState.sumCurStreamBufferLen
           << " packetsRetransmitted=" << conn_.lossState.rtxCount << " "
           << conn_;
}

void RLCongestionController::onPacketLoss(const LossEvent& loss) {
  VLOG(10) << __func__ << " lostBytes=" << loss.lostBytes
           << " lostPackets=" << loss.lostPackets
           << " cwnd=" << cwndBytes_.load() << " inflight=" << bytesInFlight_
           << " " << conn_;
  DCHECK(loss.largestLostPacketNum.hasValue());
  subtractAndCheckUnderflow(bytesInFlight_, loss.lostBytes);
  if (loss.persistentCongestion) {
    VLOG(10) << __func__ << " writable=" << getWritableBytes()
             << " cwnd=" << cwndBytes_.load() << " inflight=" << bytesInFlight_
             << " " << conn_;
  }
}

void RLCongestionController::onUpdate(const uint64_t& cwndBytes) noexcept {
  cwndBytes_ = cwndBytes;
}

void RLCongestionController::onReset() noexcept {
  // Reset to init cwnd
  cwndBytes_ = conn_.transportSettings.initCwndInMss * conn_.udpSendPacketLen;
  // TODO (viswanath): Need to flush so that the observations are reset too
  // given async env?
}

void RLCongestionController::setObservation(
    const folly::Optional<AckEvent>& ack,
    const folly::Optional<LossEvent>& loss, Observation& obs) {
  const auto& rttMin = minRTTFilter_.GetBest();
  const float rttMinMs = rttMin.count() / 1000.0;
  const float rttStandingMs = standingRTTFilter_.GetBest().count() / 1000.0;
  const float delayMs =
      duration_cast<microseconds>(conn_.lossState.lrtt - rttMin).count() /
      1000.0;

  obs[Field::RTT_MIN_MS] = rttMinMs;
  obs[Field::RTT_STANDING_MS] = rttStandingMs;
  obs[Field::LRTT_MS] = conn_.lossState.lrtt.count() / 1000.0;
  obs[Field::SRTT_MS] = conn_.lossState.srtt.count() / 1000.0;
  obs[Field::RTT_VAR_MS] = conn_.lossState.rttvar.count() / 1000.0;
  obs[Field::DELAY_MS] = delayMs;

  obs[Field::CWND_BYTES] = cwndBytes_;
  obs[Field::BYTES_IN_FLIGHT] = bytesInFlight_;
  obs[Field::WRITABLE_BYTES] = getWritableBytes();
  obs[Field::BYTES_SENT] = conn_.lossState.totalBytesSent - prevTotalBytesSent_;
  obs[Field::BYTES_RECEIVED] =
      conn_.lossState.totalBytesRecvd - prevTotalBytesRecvd_;
  obs[Field::BYTES_RETRANSMITTED] =
      conn_.lossState.totalBytesRetransmitted - prevTotalBytesRetransmitted_;

  obs[Field::PTO_COUNT] = conn_.lossState.ptoCount;
  obs[Field::TOTAL_PTO_DELTA] =
      conn_.lossState.totalPTOCount - prevTotalPTOCount_;
  obs[Field::RTX_COUNT] = conn_.lossState.rtxCount - prevRtxCount_;
  obs[Field::TIMEOUT_BASED_RTX_COUNT] =
      conn_.lossState.timeoutBasedRtxCount - prevTimeoutBasedRtxCount_;

  if (ack && ack->largestAckedPacket.hasValue()) {
    obs[Field::ACKED_BYTES] = ack->ackedBytes;
    obs[Field::ACKED_PACKETS] = ack->ackedPackets.size();
    // Calculate throughput in bytes per sec
    obs[Field::THROUGHPUT] = ack->ackedBytes * 1000.0 / rttStandingMs;
  }

  if (loss) {
    obs[Field::LOST_BYTES] = loss->lostBytes;
    obs[Field::LOST_PACKETS] = loss->lostPackets;
    obs[Field::PERSISTENT_CONGESTION] = loss->persistentCongestion;
  }

  // Update prev state values
  prevTotalBytesSent_ = conn_.lossState.totalBytesSent;
  prevTotalBytesRecvd_ = conn_.lossState.totalBytesRecvd;
  prevTotalBytesRetransmitted_ = conn_.lossState.totalBytesRetransmitted;
  prevTotalPTOCount_ = conn_.lossState.totalPTOCount;
  prevRtxCount_ = conn_.lossState.rtxCount;
  prevTimeoutBasedRtxCount_ = conn_.lossState.timeoutBasedRtxCount;
}

uint64_t RLCongestionController::getWritableBytes() const noexcept {
  const uint64_t cwndBytes = cwndBytes_.load();
  if (bytesInFlight_ > cwndBytes) {
    return 0;
  } else {
    return cwndBytes - bytesInFlight_;
  }
}

uint64_t RLCongestionController::getCongestionWindow() const noexcept {
  return cwndBytes_.load();
}

CongestionControlType RLCongestionController::type() const noexcept {
  // TODO: Update mvfst with new cc type. Just return None for now.
  return CongestionControlType::None;
}

void RLCongestionController::setConnectionEmulation(uint8_t) noexcept {}

bool RLCongestionController::canBePaced() const noexcept {
  // TODO: Think about pacing. Not supported for now.
  return false;
}

uint64_t RLCongestionController::getBytesInFlight() const noexcept {
  return bytesInFlight_;
}

uint64_t RLCongestionController::getPacingRate(
    TimePoint /* currentTime */) noexcept {
  // Pacing is not supported currently
  return conn_.transportSettings.writeConnectionDataPacketsLimit;
}

void RLCongestionController::markPacerTimeoutScheduled(
    TimePoint /* currentTime*/) noexcept { /* unsupported */
}

std::chrono::microseconds RLCongestionController::getPacingInterval() const
    noexcept {
  // Pacing is not supported currently
  return std::chrono::microseconds(
      folly::HHWheelTimerHighRes::DEFAULT_TICK_INTERVAL);
}

void RLCongestionController::setMinimalPacingInterval(
    std::chrono::microseconds interval) noexcept { /* unsupported */
}

void RLCongestionController::setAppIdle(bool,
                                        TimePoint) noexcept { /* unsupported */
}

void RLCongestionController::setAppLimited() { /* unsupported */
}

bool RLCongestionController::isAppLimited() const noexcept {
  return false;  // not supported
}

}  // namespace quic
