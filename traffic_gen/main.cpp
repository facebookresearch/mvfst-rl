/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include <glog/logging.h>

#include <fizz/crypto/Utils.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>
#include <quic/QuicConstants.h>
#include <quic/congestion_control/CongestionControllerFactory.h>
#include <quic/congestion_control/Copa.h>

#include "congestion_control/RLCongestionControllerFactory.h"
#include "traffic_gen/ExampleClient.h"
#include "traffic_gen/ExampleServer.h"

DEFINE_string(host, "::1", "Server hostname/IP");
DEFINE_int32(port, 6666, "Server port");
DEFINE_string(mode, "server", "Mode to run in: 'client' or 'server'");
DEFINE_int32(chunk_size, 1024 * 1024, "Chunk size to send at once");
DEFINE_string(cc_algo, "cubic", "Congestion Control algorithm to use");
DEFINE_string(
    cc_env_mode, "local",
    "CongestionControlEnv mode for RL cc_algo - [local|remote|random|fixed]");
DEFINE_int64(cc_env_actor_id, 0,
             "For use in training to uniquely identify an actor across "
             "episodic connections to RL server.");
DEFINE_int64(cc_env_job_id, -1,
             "Index of the current job in the list of active jobs. -1 if undefined.");
DEFINE_string(cc_env_model_file, "traced_model.pt",
              "PyTorch traced model file for local mode");
DEFINE_string(
    cc_env_rpc_address, "unix:/tmp/rl_server_path",
    "CongestionControlRPCEnv RL server address for training. Could "
    "be either <host>:<port> or Unix domain socket path unix:<path>.");
DEFINE_string(cc_env_agg, "time", "State aggregation type for RL cc_algo");
DEFINE_int32(cc_env_time_window_ms, 100,
             "Window duration (ms) for TIME_WINDOW aggregation");
DEFINE_int32(cc_env_fixed_window_size, 10,
             "Window size for FIXED_WINDOW aggregation");
DEFINE_bool(cc_env_use_state_summary, true,
            "Whether to use state summary instead of raw states in "
            "observation (auto-enabled for TIME_WINDOW)");
DEFINE_int32(
    cc_env_history_size, 2,
    "Length of history (such as past actions) to include in observation");
DEFINE_double(
    cc_env_norm_ms, 100.0,
    "Normalization factor for temporal (in ms) fields in observation");
DEFINE_double(cc_env_norm_bytes, 1000.0,
              "Normalization factor for byte fields in observation");
DEFINE_string(cc_env_actions, "0,/2,-10,+10,*2",
              "List of actions specifying how cwnd should be updated. The "
              "first action is required to be 0 (no-op action).");
DEFINE_bool(
    cc_env_reward_log_ratio, false,
    "If true, then instead of "
    " a * throughput - b * delay - c * loss "
    "we use as reward "
    " a * log(a' + throughput) - b * log(b' + delay) - c * log(c' + loss)");
DEFINE_double(cc_env_reward_throughput_factor, 0.1,
              "Throughput multiplier in reward (a)");
DEFINE_double(cc_env_reward_throughput_log_offset, 1.0,
              "Offset to add to throughput in log version (a')");
DEFINE_double(cc_env_reward_delay_factor, 0.01,
              "Delay multiplier in reward (b)");
DEFINE_double(cc_env_reward_delay_log_offset, 1.0,
              "Offset to add to delay in log version (b')");
DEFINE_double(cc_env_reward_packet_loss_factor, 0.0,
              "Packet loss multiplier in reward (c)");
DEFINE_double(cc_env_reward_packet_loss_log_offset, 1.0,
              "Offset to add to packet loss in log version (c')");
DEFINE_bool(cc_env_reward_max_delay, true,
            "Whether to take max delay over observations in reward."
            "Otherwise, avg delay is used.");
DEFINE_uint32(cc_env_fixed_cwnd, 10,
              "Target fixed cwnd value (only used in 'fixed' env mode)");
DEFINE_uint64(cc_env_min_rtt_window_length_us,
              quic::kMinRTTWindowLength.count(),
              "Window length (in us) of min RTT filter used to estimate delay");

using namespace quic::traffic_gen;
using Config = quic::CongestionControlEnv::Config;

std::shared_ptr<quic::CongestionControllerFactory>
makeRLCongestionControllerFactory() {
  Config cfg;

  if (FLAGS_cc_env_mode == "local") {
    cfg.mode = Config::Mode::LOCAL;
  } else if (FLAGS_cc_env_mode == "remote") {
    cfg.mode = Config::Mode::REMOTE;
  } else if (FLAGS_cc_env_mode == "random") {
    cfg.mode = Config::Mode::RANDOM;
  } else if (FLAGS_cc_env_mode == "fixed") {
    cfg.mode = Config::Mode::FIXED;
  } else {
    LOG(FATAL) << "Unknown cc_env_mode: " << FLAGS_cc_env_mode;
  }

  cfg.modelFile = FLAGS_cc_env_model_file;
  cfg.rpcAddress = FLAGS_cc_env_rpc_address;
  cfg.actorId = FLAGS_cc_env_actor_id;
  cfg.jobId = FLAGS_cc_env_job_id;

  if (FLAGS_cc_env_agg == "time") {
    cfg.aggregation = Config::Aggregation::TIME_WINDOW;
  } else if (FLAGS_cc_env_agg == "fixed") {
    cfg.aggregation = Config::Aggregation::FIXED_WINDOW;
  } else {
    LOG(FATAL) << "Unknown cc_env_agg: " << FLAGS_cc_env_agg;
  }
  cfg.windowDuration = std::chrono::milliseconds(FLAGS_cc_env_time_window_ms);
  cfg.windowSize = FLAGS_cc_env_fixed_window_size;
  cfg.useStateSummary = FLAGS_cc_env_use_state_summary;

  cfg.historySize = FLAGS_cc_env_history_size;

  cfg.normMs = FLAGS_cc_env_norm_ms;
  cfg.normBytes = FLAGS_cc_env_norm_bytes;

  cfg.parseActionsFromString(FLAGS_cc_env_actions);

  cfg.rewardLogRatio = FLAGS_cc_env_reward_log_ratio;
  cfg.throughputFactor = FLAGS_cc_env_reward_throughput_factor;
  cfg.throughputLogOffset = FLAGS_cc_env_reward_throughput_log_offset;
  cfg.delayFactor = FLAGS_cc_env_reward_delay_factor;
  cfg.delayLogOffset = FLAGS_cc_env_reward_delay_log_offset;
  cfg.packetLossFactor = FLAGS_cc_env_reward_packet_loss_factor;
  cfg.packetLossLogOffset = FLAGS_cc_env_reward_packet_loss_log_offset;
  cfg.maxDelayInReward = FLAGS_cc_env_reward_max_delay;
  cfg.fixedCwnd = FLAGS_cc_env_fixed_cwnd;
  cfg.minRTTWindowLength =
      std::chrono::microseconds(FLAGS_cc_env_min_rtt_window_length_us);

  auto envFactory = std::make_shared<quic::CongestionControlEnvFactory>(cfg);
  return std::make_shared<quic::RLCongestionControllerFactory>(envFactory);
}

int main(int argc, char *argv[]) {
#if FOLLY_HAVE_LIBGFLAGS
  // Enable glog logging to stderr by default.
  gflags::SetCommandLineOptionWithMode("logtostderr", "1",
                                       gflags::SET_FLAGS_DEFAULT);
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  folly::Init init(&argc, &argv);
  fizz::CryptoUtils::init();

  quic::CongestionControlType cc_algo;
  std::shared_ptr<quic::CongestionControllerFactory> ccFactory =
      std::make_shared<quic::DefaultCongestionControllerFactory>();
  if (FLAGS_cc_algo == "cubic") {
    cc_algo = quic::CongestionControlType::Cubic;
  } else if (FLAGS_cc_algo == "newreno") {
    cc_algo = quic::CongestionControlType::NewReno;
  } else if (FLAGS_cc_algo == "copa") {
    cc_algo = quic::CongestionControlType::Copa;
  } else if (FLAGS_cc_algo == "bbr") {
    cc_algo = quic::CongestionControlType::BBR;
  } else if (FLAGS_cc_algo == "rl") {
    cc_algo = quic::CongestionControlType::None;
    ccFactory = makeRLCongestionControllerFactory();
  } else if (FLAGS_cc_algo == "none") {
    cc_algo = quic::CongestionControlType::None;
  } else {
    LOG(ERROR) << "Unknown cc_algo " << FLAGS_cc_algo;
    return -1;
  }

  if (FLAGS_mode == "server") {
    ExampleServer server(FLAGS_host, FLAGS_port, cc_algo, ccFactory);
    server.start();
  } else if (FLAGS_mode == "client") {
    if (FLAGS_host.empty() || FLAGS_port == 0) {
      LOG(ERROR) << "ExampleClient expected --host and --port";
      return -2;
    }
    ExampleClient client(FLAGS_host, FLAGS_port, cc_algo, ccFactory);
    client.start();
  } else {
    LOG(ERROR) << "Unknown mode specified: " << FLAGS_mode;
    return -1;
  }
  return 0;
}
