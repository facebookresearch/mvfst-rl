#pragma once

#include <grpc++/grpc++.h>
#include <condition_variable>
#include <thread>

#include "rpcenv.grpc.pb.h"
#include "rpcenv.pb.h"

#include "CongestionControlEnv.h"

namespace quic {

class CongestionControlRPCEnv : public CongestionControlEnv,
                                public rpcenv::RPCEnvServer::Service {
 private:
  class EnvServer {
   public:
    EnvServer(rpcenv::RPCEnvServer::Service* service, int port)
        : service_(CHECK_NOTNULL(service)), port_(port) {}

    void start() {
      if (thread_) {
        LOG(WARNING) << "RPCEnvServer already running";
        return;
      }

      // CV to wait until server fully starts
      std::mutex mutex;
      std::condition_variable cv;
      bool started = false;

      // Server needs to be started on a different thread as it blocks
      // waiting for incoming connections.
      thread_ = std::make_unique<std::thread>([&] {
        std::string server_address("0.0.0.0:" + std::to_string(port_));
        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_address,
                                 grpc::InsecureServerCredentials());
        builder.RegisterService(service_);
        server_ = builder.BuildAndStart();

        {
          std::lock_guard<std::mutex> g(mutex);
          started = true;
        }
        cv.notify_one();

        LOG(INFO) << "RPCEnvServer listening on " << server_address;
        server_->Wait();
      });

      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&]() -> bool { return started; });
    }

    void stop() {
      if (thread_) {
        server_->Shutdown();
        thread_->join();
      }
      server_.reset();
      thread_.reset();
    }

   private:
    rpcenv::RPCEnvServer::Service* service_{nullptr};
    int port_;
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<std::thread> thread_;  // Thread to start the server in
  };

 public:
  // TODO (viswanath): Configure port
  CongestionControlRPCEnv(CongestionControlEnv::Callback* cob,
                          int port = 60000);
  ~CongestionControlRPCEnv() override;

 private:
  void onObservation(const std::vector<Observation>& observations) override;

  grpc::Status StreamingEnv(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<rpcenv::Step, rpcenv::Action>* stream) override;

  static void fillNDArray(rpcenv::NDArray* ndarray,
                          const torch::Tensor& tensor);

  std::unique_ptr<EnvServer> envServer_;
  std::atomic<bool> shutdown_{false};  // Signals termination of env loop

  torch::Tensor tensor_;  // Tensor holding observations
  float reward_;
  bool observationReady_{false};

  // CV to protect tensor_, reward_ and observationReady_, and signal grpc
  // thread when state update is ready to be sent.
  std::condition_variable cv_;
  std::mutex mutex_;
};

}  // namespace quic
