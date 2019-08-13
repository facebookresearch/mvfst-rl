#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace quic {

struct CongestionControlEnvConfig {
  /// Definitions

  enum class Mode : uint8_t {
    TRAIN = 0,
    TEST,
    RANDOM,  // Env that takes random actions
  };

  // Type of aggregation to group state updates
  enum class Aggregation : uint8_t {
    TIME_WINDOW = 0,  // Group state updates every X ms
    FIXED_WINDOW,     // Group every Y state updates
    // TODO: Other kinds of aggregation (like avg/max/ewma)?
  };

  enum class ActionOp : uint8_t {
    NOOP = 0,
    ADD,
    SUB,
    MUL,
    DIV,
  };

  /// Members

  Mode mode{Mode::TRAIN};

  // RL server address ("<host>:<port>" or "unix:<path>") for RPC Env.
  std::string rpcAddress{"unix:/tmp/rl_server_path"};

  Aggregation aggregation{Aggregation::TIME_WINDOW};
  std::chrono::milliseconds windowDuration{500};  // Time window duration
  uint32_t windowSize{10};                        // Fixed window size

  // Normalization factors for observation fields
  float normMs{100.0};
  float normBytes{1000.0};

  // Number of past actions taken to include in observation
  uint32_t numPastActions{2};

  // Default actions: [noop, cwnd / 2, cwnd - 10, cwnd + 10, cwnd * 2]
  std::vector<std::pair<ActionOp, float>> actions{
      {ActionOp::NOOP, 0}, {ActionOp::DIV, 2}, {ActionOp::SUB, 10},
      {ActionOp::ADD, 10}, {ActionOp::MUL, 2},
  };

  // Multipliers for reward components
  float throughputFactor{1.0};
  float delayFactor{0.5};
  float packetLossFactor{0.0};

  // Whether to use max delay in reward (avg by default)
  bool maxDelayInReward{false};

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
