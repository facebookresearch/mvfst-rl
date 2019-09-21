/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <glog/logging.h>

#include <quic/server/QuicServer.h>
#include <quic/server/QuicServerTransport.h>
#include <quic/server/QuicSharedUDPSocketFactory.h>

#include <traffic_gen/ExampleHandler.h>
#include <traffic_gen/Utils.h>

namespace quic {
namespace traffic_gen {

class ExampleServerTransportFactory : public quic::QuicServerTransportFactory {
 public:
  ~ExampleServerTransportFactory() override {
    while (!exampleHandlers_.empty()) {
      auto& handler = exampleHandlers_.back();
      handler->getEventBase()->runImmediatelyOrRunInEventBaseThreadAndWait(
          [this] {
            // The evb should be performing a sequential consistency atomic
            // operation already, so we can bank on that to make sure the writes
            // propagate to all threads.
            exampleHandlers_.pop_back();
          });
    }
  }

  ExampleServerTransportFactory() {}

  quic::QuicServerTransport::Ptr make(
      folly::EventBase* evb, std::unique_ptr<folly::AsyncUDPSocket> sock,
      const folly::SocketAddress&,
      std::shared_ptr<const fizz::server::FizzServerContext>
          ctx) noexcept override {
    CHECK_EQ(evb, sock->getEventBase());
    auto exampleHandler = std::make_unique<ExampleHandler>(evb);
    auto transport = quic::QuicServerTransport::make(evb, std::move(sock),
                                                     *exampleHandler, ctx);
    exampleHandler->setQuicSocket(transport);
    exampleHandlers_.push_back(std::move(exampleHandler));
    return transport;
  }

  std::vector<std::unique_ptr<ExampleHandler>> exampleHandlers_;

 private:
};

class ExampleServer {
 public:
  explicit ExampleServer(
      const std::string& host = "::1", uint16_t port = 6666,
      CongestionControlType cc_algo = CongestionControlType::Cubic,
      std::shared_ptr<CongestionControllerFactory> ccFactory =
          std::make_shared<DefaultCongestionControllerFactory>())
      : host_(host), port_(port), server_(QuicServer::createQuicServer()) {
    server_->setQuicServerTransportFactory(
        std::make_unique<ExampleServerTransportFactory>());
    server_->setFizzContext(createTestServerCtx());
    server_->setCongestionControllerFactory(ccFactory);
    TransportSettings settings;
    settings.defaultCongestionController = cc_algo;
    server_->setTransportSettings(settings);
  }

  void start() {
    // Create a SocketAddress and the default or passed in host.
    folly::SocketAddress addr(host_.c_str(), port_);
    server_->start(addr, 0);
    LOG(INFO) << "ExampleServer started at: " << addr.describe();
    eventbase_.loopForever();
  }

 private:
  std::string host_;
  uint16_t port_;
  folly::EventBase eventbase_;
  std::shared_ptr<quic::QuicServer> server_;
};

}  // namespace traffic_gen
}  // namespace quic
