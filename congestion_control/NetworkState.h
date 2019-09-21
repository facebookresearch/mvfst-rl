/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

namespace quic {

struct NetworkState {
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

  static constexpr uint16_t kNumFields =
      static_cast<uint16_t>(Field::NUM_FIELDS);

  NetworkState() : data_(kNumFields, 0.0) {}

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

  torch::Tensor toTensor() const;
  void toTensor(torch::Tensor& tensor) const;
  static torch::Tensor toTensor(const std::vector<NetworkState>& states);
  static void toTensor(const std::vector<NetworkState>& states,
                       torch::Tensor& tensor);

  static std::vector<NetworkState> fromTensor(const torch::Tensor& tensor);

  static std::string fieldToString(const uint16_t field);
  static std::string fieldToString(const Field field);

 private:
  std::vector<float> data_;
};

std::ostream& operator<<(std::ostream& os, const NetworkState& state);

}  // namespace quic
