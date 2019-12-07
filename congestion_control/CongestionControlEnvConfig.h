/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace quic {

struct CongestionControlEnvConfig {
  /// Definitions

  enum class Mode : uint8_t {
    LOCAL = 0,  // RL policy run locally
    REMOTE,     // RL policy on a remote RL server
    RANDOM,     // Simple env that takes random actions (for testing)
  };

  // Type of aggregation to group state updates
  enum class Aggregation : uint8_t {
    TIME_WINDOW = 0,  // Group state updates every X ms
    FIXED_WINDOW,     // Group every Y state updates
  };

  enum class ActionOp : uint8_t {
    NOOP = 0,
    ADD,
    SUB,
    MUL,
    DIV,
  };

  /// Members

  Mode mode{Mode::LOCAL};

  // PyTorch traced model file to load for local mode
  std::string modelFile{""};

  // RL server address ("<host>:<port>" or "unix:<path>") for remote mode.
  std::string rpcAddress{"unix:/tmp/rl_server_path"};

  // For use in training to uniquely identify an actor across episodic
  // connections to RL server.
  int64_t actorId{0};

  Aggregation aggregation{Aggregation::TIME_WINDOW};
  std::chrono::milliseconds windowDuration{100};  // Time window duration
  uint32_t windowSize{10};                        // Fixed window size
  bool useStateSummary{true};  // Whether to use state summary instead of raw
                               // states (auto-enabled for TIME_WINDOW).

  // Normalization factors for observation fields
  float normMs{100.0};
  float normBytes{1000.0};

  // Size of history (such as past actions) to include in observation
  uint32_t historySize{2};

  // Default actions: [noop, cwnd / 2, cwnd - 10, cwnd + 10, cwnd * 2]
  std::vector<std::pair<ActionOp, float>> actions{
      {ActionOp::NOOP, 0}, {ActionOp::DIV, 2}, {ActionOp::SUB, 10},
      {ActionOp::ADD, 10}, {ActionOp::MUL, 2},
  };

  // Multipliers for reward components
  float throughputFactor{0.1};
  float delayFactor{0.01};
  float packetLossFactor{0.0};

  // Whether to use max delay within a window in reward (avg otherwise)
  bool maxDelayInReward{true};

  /// Helper functions

  /**
   * Actions should be specified as string of comma-separated items of the
   * format "<op><val>". <op> can be one of [+, -, *, /]. <val> can be any
   * float. The first item should be "0", which means NOOP action.
   *
   * Example: "0,/2,-10,+10,*2".
   */
  void parseActionsFromString(const std::string& actionsStr);
  static ActionOp charToActionOp(const char op);
};

}  // namespace quic
