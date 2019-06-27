#pragma once

#include <iostream>
#include <string>
#include <thread>

#include <folly/io/async/ScopedEventBaseThread.h>
#include <glog/logging.h>
#include <quic/api/QuicSocket.h>
#include <quic/client/QuicClientTransport.h>
#include <quic/congestion_control/CongestionControllerFactory.h>

#include <traffic_gen/Utils.h>

namespace quic {
namespace traffic_gen {

class ExampleClient : public quic::QuicSocket::ConnectionCallback,
                      public quic::QuicSocket::ReadCallback,
                      public quic::QuicSocket::WriteCallback,
                      public quic::QuicSocket::DataExpiredCallback {
 public:
  ExampleClient(const std::string& host, uint16_t port,
                CongestionControlType cc_algo = CongestionControlType::Cubic,
                std::shared_ptr<CongestionControllerFactory> ccFactory =
                    std::make_shared<DefaultCongestionControllerFactory>())
      : host_(host), port_(port), cc_algo_(cc_algo), ccFactory_(ccFactory) {}

  void readAvailable(quic::StreamId streamId) noexcept override {
    auto readData = quicClient_->read(streamId, 0);
    if (readData.hasError()) {
      LOG(ERROR) << "ExampleClient failed read from stream=" << streamId
                 << ", error=" << (uint32_t)readData.error();
    }
    auto copy = readData->first->clone();
    if (recvOffsets_.find(streamId) == recvOffsets_.end()) {
      recvOffsets_[streamId] = copy->length();
    } else {
      recvOffsets_[streamId] += copy->length();
    }
    VLOG(2) << "Client received data=" << copy->computeChainDataLength()
            << " bytes on stream=" << streamId;
  }

  void readError(
      quic::StreamId streamId,
      std::pair<quic::QuicErrorCode, folly::Optional<folly::StringPiece>>
          error) noexcept override {
    LOG(ERROR) << "ExampleClient failed read from stream=" << streamId
               << ", error=" << toString(error);
    // A read error only terminates the ingress portion of the stream state.
    // Your application should probably terminate the egress portion via
    // resetStream
  }

  void onNewBidirectionalStream(quic::StreamId id) noexcept override {
    LOG(INFO) << "ExampleClient: new bidirectional stream=" << id;
    quicClient_->setReadCallback(id, this);
  }

  void onNewUnidirectionalStream(quic::StreamId id) noexcept override {
    LOG(INFO) << "ExampleClient: new unidirectional stream=" << id;
    quicClient_->setReadCallback(id, this);
  }

  void onStopSending(quic::StreamId id,
                     quic::ApplicationErrorCode /*error*/) noexcept override {
    VLOG(10) << "ExampleClient got StopSending stream id=" << id;
  }

  void onTransportReady() noexcept override {
    LOG(INFO) << "ExampleClient connected";
    auto streamId = quicClient_->createBidirectionalStream().value();
    quicClient_->setReadCallback(streamId, this);
    pendingOutput_[streamId].append(folly::IOBuf::copyBuffer("hello"));
    sendMessage(streamId, pendingOutput_[streamId]);
  }

  void onConnectionEnd() noexcept override {
    LOG(INFO) << "ExampleClient connection end";
  }

  void onConnectionError(
      std::pair<quic::QuicErrorCode, std::string> error) noexcept override {
    LOG(ERROR) << "ExampleClient error: " << toString(error.first);
  }

  void onStreamWriteReady(quic::StreamId id,
                          uint64_t maxToSend) noexcept override {
    LOG(INFO) << "ExampleClient socket is write ready with maxToSend="
              << maxToSend;
    sendMessage(id, pendingOutput_[id]);
  }

  void onStreamWriteError(
      quic::StreamId id,
      std::pair<quic::QuicErrorCode, folly::Optional<folly::StringPiece>>
          error) noexcept override {
    LOG(ERROR) << "ExampleClient write error with stream=" << id
               << " error=" << toString(error);
  }

  void onDataExpired(StreamId streamId, uint64_t newOffset) noexcept override {
    LOG(INFO) << "Client received skipData; "
              << newOffset - recvOffsets_[streamId]
              << " bytes skipped on stream=" << streamId;
  }

  void start() {
    folly::ScopedEventBaseThread networkThread("ExampleClientThread");
    evb_ = networkThread.getEventBase();
    folly::SocketAddress addr(host_.c_str(), port_);

    evb_->runInEventBaseThreadAndWait([&] {
      auto sock = std::make_unique<folly::AsyncUDPSocket>(evb_);
      quicClient_ =
          std::make_shared<quic::QuicClientTransport>(evb_, std::move(sock));
      quicClient_->setHostname("example.org");
      quicClient_->setCertificateVerifier(
          std::make_unique<DummyCertificateVerifier>());
      quicClient_->addNewPeerAddress(addr);
      quicClient_->setCongestionControllerFactory(ccFactory_);

      TransportSettings settings;
      settings.defaultCongestionController = cc_algo_;
      quicClient_->setTransportSettings(settings);

      LOG(INFO) << "ExampleClient connecting to " << addr.describe();
      quicClient_->start(this);
    });

    // Loop forever
    while (true) {
    }

    LOG(INFO) << "ExampleClient stopping client";
  }

  ~ExampleClient() override = default;

 private:
  void sendMessage(quic::StreamId id, folly::IOBufQueue& data) {
    auto message = data.move();
    auto res = quicClient_->writeChain(id, message->clone(), true, false);
    if (res.hasError()) {
      LOG(ERROR) << "ExampleClient writeChain error=" << uint32_t(res.error());
    } else if (res.value()) {
      LOG(INFO)
          << "ExampleClient socket did not accept all data, buffering len="
          << res.value()->computeChainDataLength();
      data.append(std::move(res.value()));
      quicClient_->notifyPendingWriteOnStream(id, this);
    } else {
      auto str = message->moveToFbString().toStdString();
      LOG(INFO) << "ExampleClient wrote \"" << str << "\""
                << ", len=" << str.size() << " on stream=" << id;
      // sent whole message
      pendingOutput_.erase(id);
    }
  }

  std::string host_;
  uint16_t port_;
  CongestionControlType cc_algo_;
  std::shared_ptr<CongestionControllerFactory> ccFactory_;

  std::shared_ptr<quic::QuicClientTransport> quicClient_;
  std::map<quic::StreamId, folly::IOBufQueue> pendingOutput_;
  std::map<quic::StreamId, uint64_t> recvOffsets_;
  folly::EventBase* evb_{nullptr};
};

}  // namespace traffic_gen
}  // namespace quic
