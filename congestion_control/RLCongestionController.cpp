#include <quic/common/TimeUtil.h>
#include <quic/congestion_control/CongestionControlFunctions.h>
#include <quic/congestion_control/Copa.h>

#include <congestion_control/RLCongestionController.h>

namespace quic {

using namespace std::chrono;

RLCongestionController::RLCongestionController(QuicConnectionStateBase& conn)
    : conn_(conn),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen),
      minRTTFilter_(kMinRTTWindowLength.count(), 0us, 0),
      standingRTTFilter_(100000, /*100ms*/
                         0us, 0) {
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

void RLCongestionController::onPacketSent(const OutstandingPacket& packet) {
  addAndCheckOverflow(bytesInFlight_, packet.encodedSize);

  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_
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

  // TODO (viswanath): env hook
}

void RLCongestionController::onPacketAcked(const AckEvent& ack) {
  DCHECK(ack.largestAckedPacket.hasValue());
  subtractAndCheckUnderflow(bytesInFlight_, ack.ackedBytes);
  minRTTFilter_.Update(
      conn_.lossState.lrtt,
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());
  auto rttMin = minRTTFilter_.GetBest();
  standingRTTFilter_.SetWindowLength(conn_.lossState.srtt.count() / 2);
  standingRTTFilter_.Update(
      conn_.lossState.lrtt,
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());
  auto rttStandingMicroSec = standingRTTFilter_.GetBest().count();

  VLOG(10) << __func__ << "ack size=" << ack.ackedBytes
           << " num packets acked=" << ack.ackedBytes / conn_.udpSendPacketLen
           << " writable=" << getWritableBytes() << " cwnd=" << cwndBytes_
           << " inflight=" << bytesInFlight_ << " rttMin=" << rttMin.count()
           << " sRTT=" << conn_.lossState.srtt.count()
           << " lRTT=" << conn_.lossState.lrtt.count()
           << " mRTT=" << conn_.lossState.mrtt.count()
           << " rttvar=" << conn_.lossState.rttvar.count()
           << " packetsBufferred="
           << conn_.flowControlState.sumCurStreamBufferLen
           << " packetsRetransmitted=" << conn_.lossState.rtxCount << " "
           << conn_;

  auto delayInMicroSec =
      duration_cast<microseconds>(conn_.lossState.lrtt - rttMin).count();
  if (delayInMicroSec < 0) {
    LOG(ERROR) << __func__
               << "delay negative, lrtt=" << conn_.lossState.lrtt.count()
               << " rttMin=" << rttMin.count() << " " << conn_;
    return;
  }
  if (rttStandingMicroSec == 0) {
    LOG(ERROR) << __func__ << "rttStandingMicroSec zero, lrtt = "
               << conn_.lossState.lrtt.count() << " rttMin=" << rttMin.count()
               << " " << conn_;
    return;
  }

  VLOG(10) << __func__
           << " estimated queuing delay microsec =" << delayInMicroSec << " "
           << conn_;

  // TODO (viswanath): env hook
}

void RLCongestionController::onPacketLoss(const LossEvent& loss) {
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

  // TODO (viswanath): env hook
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
  // TODO (viswanath): Update mvfst with new cc type. Just return None for now.
  return CongestionControlType::None;
}

void RLCongestionController::setConnectionEmulation(uint8_t) noexcept {}

bool RLCongestionController::canBePaced() const noexcept {
  // TODO (viswanath): Think about pacing. Not supported for now.
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
