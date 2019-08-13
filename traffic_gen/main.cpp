#include <glog/logging.h>

#include <fizz/crypto/Utils.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>
#include <quic/QuicConstants.h>
#include <quic/congestion_control/CongestionControllerFactory.h>

#include "congestion_control/RLCongestionControllerFactory.h"
#include "traffic_gen/ExampleClient.h"
#include "traffic_gen/ExampleServer.h"

DEFINE_string(host, "::1", "Server hostname/IP");
DEFINE_int32(port, 6666, "Server port");
DEFINE_string(mode, "server", "Mode to run in: 'client' or 'server'");
DEFINE_int32(chunk_size, 64 * 1024, "Chunk size to send at once");
DEFINE_string(cc_algo, "cubic", "Congestion Control algorithm to use");
DEFINE_string(cc_env_mode, "test",
              "CongestionControlEnv mode for RL cc_algo - [train|test|random]");
DEFINE_string(
    cc_env_rpc_address, "unix:/tmp/rl_server_path",
    "CongestionControlRPCEnv RL server address for training. Could "
    "be either <host>:<port> or Unix domain socket path unix:<path>.");
DEFINE_string(cc_env_agg, "time", "State aggregation type for RL cc_algo");
DEFINE_int32(cc_env_time_window_ms, 500,
             "Window duration (ms) for TIME_WINDOW aggregation");
DEFINE_int32(cc_env_fixed_window_size, 10,
             "Window size for FIXED_WINDOW aggregation");
DEFINE_int32(cc_env_num_past_actions, 2,
             "Number of past actions taken to include in observation");
DEFINE_double(
    cc_env_norm_ms, 100.0,
    "Normalization factor for temporal (in ms) fields in observation");
DEFINE_double(cc_env_norm_bytes, 1000.0,
              "Normalization factor for byte fields in observation");
DEFINE_string(cc_env_actions, "0,/2,-10,+10,*2",
              "List of actions specifying how cwnd should be updated. The "
              "first action is required to be 0 (no-op action).");
DEFINE_double(cc_env_reward_throughput_factor, 1.0,
              "Throughput multiplier in reward");
DEFINE_double(cc_env_reward_delay_factor, 0.5, "Delay multiplier in reward");
DEFINE_double(cc_env_reward_packet_loss_factor, 0.0,
              "Packet loss multiplier in reward");
DEFINE_bool(cc_env_reward_max_delay, false,
            "Whether to take max delay over observations in reward."
            "By default, avg delay is used.");

using namespace quic::traffic_gen;
using Config = quic::CongestionControlEnv::Config;

std::shared_ptr<quic::CongestionControllerFactory>
makeRLCongestionControllerFactory() {
  Config cfg;

  if (FLAGS_cc_env_mode == "train") {
    cfg.mode = Config::Mode::TRAIN;
  } else if (FLAGS_cc_env_mode == "test") {
    cfg.mode = Config::Mode::TEST;
  } else if (FLAGS_cc_env_mode == "random") {
    cfg.mode = Config::Mode::RANDOM;
  } else {
    LOG(FATAL) << "Unknown cc_env_mode: " << FLAGS_cc_env_mode;
  }

  cfg.rpcAddress = FLAGS_cc_env_rpc_address;

  if (FLAGS_cc_env_agg == "time") {
    cfg.aggregation = Config::Aggregation::TIME_WINDOW;
  } else if (FLAGS_cc_env_agg == "fixed") {
    cfg.aggregation = Config::Aggregation::FIXED_WINDOW;
  } else {
    LOG(FATAL) << "Unknown cc_env_agg: " << FLAGS_cc_env_agg;
  }
  cfg.windowDuration = std::chrono::milliseconds(FLAGS_cc_env_time_window_ms);
  cfg.windowSize = FLAGS_cc_env_fixed_window_size;

  cfg.numPastActions = FLAGS_cc_env_num_past_actions;

  cfg.normMs = FLAGS_cc_env_norm_ms;
  cfg.normBytes = FLAGS_cc_env_norm_bytes;

  cfg.parseActionsFromString(FLAGS_cc_env_actions);

  cfg.throughputFactor = FLAGS_cc_env_reward_throughput_factor;
  cfg.delayFactor = FLAGS_cc_env_reward_delay_factor;
  cfg.packetLossFactor = FLAGS_cc_env_reward_packet_loss_factor;
  cfg.maxDelayInReward = FLAGS_cc_env_reward_max_delay;

  auto envFactory = std::make_shared<quic::CongestionControlEnvFactory>(cfg);
  return std::make_shared<quic::RLCongestionControllerFactory>(envFactory);
}

int main(int argc, char* argv[]) {
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
    // TODO: Update cc_algo type
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
