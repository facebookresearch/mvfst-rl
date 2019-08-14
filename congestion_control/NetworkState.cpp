#include "NetworkState.h"

#include <glog/logging.h>

namespace quic {

std::string NetworkState::fieldToString(const uint16_t field) {
  return fieldToString(static_cast<Field>(field));
}

std::string NetworkState::fieldToString(const Field field) {
  switch (field) {
    case Field::RTT_MIN:
      return "RTT_MIN";
    case Field::RTT_STANDING:
      return "RTT_STANDING";
    case Field::LRTT:
      return "LRTT";
    case Field::SRTT:
      return "SRTT";
    case Field::RTT_VAR:
      return "RTT_VAR";
    case Field::DELAY:
      return "DELAY";
    case Field::CWND:
      return "CWND";
    case Field::IN_FLIGHT:
      return "IN_FLIGHT";
    case Field::WRITABLE:
      return "WRITABLE";
    case Field::SENT:
      return "SENT";
    case Field::RECEIVED:
      return "RECEIVED";
    case Field::RETRANSMITTED:
      return "RETRANSMITTED";
    case Field::PTO_COUNT:
      return "PTO_COUNT";
    case Field::TOTAL_PTO_DELTA:
      return "TOTAL_PTO_DELTA";
    case Field::RTX_COUNT:
      return "RTX_COUNT";
    case Field::TIMEOUT_BASED_RTX_COUNT:
      return "TIMEOUT_BASED_RTX_COUNT";
    case Field::ACKED:
      return "ACKED";
    case Field::THROUGHPUT:
      return "THROUGHPUT";
    case Field::LOST:
      return "LOST";
    case Field::PERSISTENT_CONGESTION:
      return "PERSISTENT_CONGESTION";
    case Field::NUM_FIELDS:
      return "NUM_FIELDS";
    default:
      LOG(FATAL) << "Unknown field";
      break;
  }
  __builtin_unreachable();
}

std::ostream& operator<<(std::ostream& os, const NetworkState& state) {
  os << "NetworkState (" << state.size() << " fields): ";
  for (size_t i = 0; i < state.size(); ++i) {
    os << i << ". " << NetworkState::fieldToString(i) << "=" << state[i] << " ";
  }
  return os;
}

}  // namespace quic
