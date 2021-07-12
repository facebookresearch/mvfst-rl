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
DEFINE_int64(cc_env_job_count, -1,
             "Job counter during training. -1 if undefined.");
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
DEFINE_double(cc_env_uplink_bandwidth, 0.0,
              "Maximum bandwidth (in MBytes/s) achievable by the uplink");
DEFINE_int32(cc_env_uplink_queue_size_bytes, 1,
             "Size of the uplink queue (in bytes)");
DEFINE_double(
    cc_env_base_rtt, 1.0,
    "Minimum RTT that can be achieved based on network settings (in ms)");
DEFINE_string(cc_env_reward_formula, "log_ratio",
              "Which formula to use for the reward, among: "
              "linear, log_ratio, min_throughput "
              "(see pantheon_env.py for details)");
DEFINE_double(cc_env_reward_delay_offset, 0.1,
              "Offset to remove from the delay when computing the reward (o)");
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
DEFINE_double(cc_env_reward_min_throughput_ratio, 0.9,
              "Min ratio of the maximum achievable throughput / target cwnd "
              "that we want to reach (r).");
DEFINE_double(cc_env_reward_max_throughput_ratio, 1.0,
              "Max ratio of the maximum achievable throughput / target cwnd "
              "that we want to reach (r).");
DEFINE_double(
    cc_env_reward_n_packets_offset, 1.0,
    "Offset to add to the estimated number of packets in the queue (k).");
DEFINE_double(cc_env_reward_uplink_queue_max_fill_ratio, 0.5,
              "We allow the uplink queue to be filled up to this ratio without "
              "penalty (f)");
DEFINE_bool(cc_env_reward_max_delay, true,
            "Whether to take max delay over observations in reward."
            "Otherwise, avg delay is used.");
DEFINE_uint32(cc_env_fixed_cwnd, 10,
              "Target fixed cwnd value (only used in 'fixed' env mode)");
DEFINE_uint64(cc_env_min_rtt_window_length_us,
              quic::kMinRTTWindowLength.count(),
              "Window length (in us) of min RTT filter used to estimate delay");
DEFINE_double(
    cc_env_ack_delay_avg_coeff, 0.1,
    "Moving average coefficient used to compute the average ACK delay, as "
    "well as the average total RTT (including ACK delay). "
    "This is the weight of new observations: higher values update the average "
    "faster.");
DEFINE_uint32(cc_env_bandwidth_min_window_duration_ms, 100,
              "Minimum duration over which the bandwidth is computed (in ms)");
DEFINE_double(cc_env_rtt_noise_std, 0.,
              "Standard deviation of the noise added to RTT measurements");
DEFINE_double(cc_env_obs_scaling, 1.0,
              "Factor by which the observation vector should be scaled");
DEFINE_string(cc_env_stats_file, "",
              "Path to file where some statistics are saved (if provided)");

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
  cfg.jobCount = FLAGS_cc_env_job_count;

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

  if (FLAGS_cc_env_reward_formula == "linear") {
    cfg.rewardFormula = Config::RewardFormula::LINEAR;
  } else if (FLAGS_cc_env_reward_formula == "log_ratio") {
    cfg.rewardFormula = Config::RewardFormula::LOG_RATIO;
  } else if (FLAGS_cc_env_reward_formula == "min_throughput") {
    cfg.rewardFormula = Config::RewardFormula::MIN_THROUGHPUT;
  } else if (FLAGS_cc_env_reward_formula == "target_cwnd") {
    cfg.rewardFormula = Config::RewardFormula::TARGET_CWND;
  } else if (FLAGS_cc_env_reward_formula == "target_cwnd_shaped") {
    cfg.rewardFormula = Config::RewardFormula::TARGET_CWND_SHAPED;
  } else if (FLAGS_cc_env_reward_formula == "higher_is_better") {
    cfg.rewardFormula = Config::RewardFormula::HIGHER_IS_BETTER;
  } else if (FLAGS_cc_env_reward_formula == "above_cwnd") {
    cfg.rewardFormula = Config::RewardFormula::ABOVE_CWND;
  } else if (FLAGS_cc_env_reward_formula == "cwnd_range") {
    cfg.rewardFormula = Config::RewardFormula::CWND_RANGE;
  } else if (FLAGS_cc_env_reward_formula == "cwnd_range_soft") {
    cfg.rewardFormula = Config::RewardFormula::CWND_RANGE_SOFT;
  } else if (FLAGS_cc_env_reward_formula == "cwnd_tradeoff") {
    cfg.rewardFormula = Config::RewardFormula::CWND_TRADEOFF;
  } else if (FLAGS_cc_env_reward_formula == "below_target_cwnd") {
    cfg.rewardFormula = Config::RewardFormula::BELOW_TARGET_CWND;
  } else {
    LOG(FATAL) << "Unknown cc_env_reward_formula: "
               << FLAGS_cc_env_reward_formula;
  }

  cfg.uplinkBandwidth = FLAGS_cc_env_uplink_bandwidth;
  cfg.uplinkQueueSizeBytes = FLAGS_cc_env_uplink_queue_size_bytes;
  cfg.baseRTT = FLAGS_cc_env_base_rtt;
  cfg.delayOffset = FLAGS_cc_env_reward_delay_offset;
  cfg.throughputFactor = FLAGS_cc_env_reward_throughput_factor;
  cfg.throughputLogOffset = FLAGS_cc_env_reward_throughput_log_offset;
  cfg.delayFactor = FLAGS_cc_env_reward_delay_factor;
  cfg.delayLogOffset = FLAGS_cc_env_reward_delay_log_offset;
  cfg.packetLossFactor = FLAGS_cc_env_reward_packet_loss_factor;
  cfg.packetLossLogOffset = FLAGS_cc_env_reward_packet_loss_log_offset;
  cfg.minThroughputRatio = FLAGS_cc_env_reward_min_throughput_ratio;
  cfg.maxThroughputRatio = FLAGS_cc_env_reward_max_throughput_ratio;
  cfg.nPacketsOffset = FLAGS_cc_env_reward_n_packets_offset;
  cfg.uplinkQueueMaxFillRatio = FLAGS_cc_env_reward_uplink_queue_max_fill_ratio;
  cfg.maxDelayInReward = FLAGS_cc_env_reward_max_delay;
  cfg.fixedCwnd = FLAGS_cc_env_fixed_cwnd;
  cfg.minRTTWindowLength =
      std::chrono::microseconds(FLAGS_cc_env_min_rtt_window_length_us);
  cfg.rttNoiseStd = FLAGS_cc_env_rtt_noise_std;
  cfg.ackDelayAvgCoeff = FLAGS_cc_env_ack_delay_avg_coeff;
  cfg.bandwidthMinWindowDuration =
      std::chrono::milliseconds(FLAGS_cc_env_bandwidth_min_window_duration_ms);
  cfg.obsScaling = FLAGS_cc_env_obs_scaling;
  cfg.statsFile = FLAGS_cc_env_stats_file;

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
