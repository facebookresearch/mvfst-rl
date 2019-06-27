#include <glog/logging.h>

#include <fizz/crypto/Utils.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>
#include <quic/QuicConstants.h>
#include <quic/congestion_control/CongestionControllerFactory.h>

#include <congestion_control/RLCongestionControllerFactory.h>
#include <traffic_gen/ExampleClient.h>
#include <traffic_gen/ExampleServer.h>

DEFINE_string(host, "::1", "Server hostname/IP");
DEFINE_int32(port, 6666, "Server port");
DEFINE_string(mode, "server", "Mode to run in: 'client' or 'server'");
DEFINE_int32(chunk_size, 64 * 1024, "Chunk size to send at once");
DEFINE_string(cc_algo, "cubic", "Congestion Control algorithm to use");

using namespace quic::traffic_gen;

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
    // TODO (viswanath): Update cc type
    cc_algo = quic::CongestionControlType::None;
    ccFactory = std::make_shared<quic::RLCongestionControllerFactory>();
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
