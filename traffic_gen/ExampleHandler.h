/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#pragma once

#include <quic/api/QuicSocket.h>

#include <folly/io/IOBufQueue.h>
#include <folly/io/async/EventBase.h>
#include <folly/portability/GFlags.h>

DECLARE_int32(chunk_size);

namespace quic {
namespace traffic_gen {

class ExampleHandler : public quic::QuicSocket::ConnectionCallback,
                       public quic::QuicSocket::ReadCallback,
                       public quic::QuicSocket::WriteCallback {
 public:
  using StreamData = std::pair<folly::IOBufQueue, bool>;

  explicit ExampleHandler(folly::EventBase* evbIn) : evb(evbIn) {
    // Create dummy data to send
    std::string data(FLAGS_chunk_size, 'x');
    respBuf_ = folly::IOBuf::copyBuffer(data);
  }

  void setQuicSocket(std::shared_ptr<quic::QuicSocket> socket) {
    sock = socket;
  }

  void onNewBidirectionalStream(quic::StreamId id) noexcept override {
    LOG(INFO) << "Got bidirectional stream id=" << id;
    sock->setReadCallback(id, this);
  }

  void onNewUnidirectionalStream(quic::StreamId id) noexcept override {
    LOG(INFO) << "Got unidirectional stream id=" << id;
    sock->setReadCallback(id, this);
  }

  void onStopSending(quic::StreamId id,
                     quic::ApplicationErrorCode error) noexcept override {
    LOG(INFO) << "Got StopSending stream id=" << id << " error=" << error;
  }

  void onConnectionEnd() noexcept override { LOG(INFO) << "Socket closed"; }

  void onConnectionError(
      std::pair<quic::QuicErrorCode, std::string> error) noexcept override {
    LOG(ERROR) << "Socket error=" << toString(error.first);
  }

  void readAvailable(quic::StreamId id) noexcept override {
    LOG(INFO) << "read available for stream id=" << id;

    auto res = sock->read(id, 0);
    if (res.hasError()) {
      LOG(ERROR) << "Got error=" << toString(res.error());
      return;
    }
    if (input_.find(id) == input_.end()) {
      input_.emplace(
          id,
          std::make_pair(
              folly::IOBufQueue(folly::IOBufQueue::cacheChainLength()), false));
    }
    quic::Buf data = std::move(res.value().first);
    bool eof = res.value().second;
    auto dataLen = (data ? data->computeChainDataLength() : 0);
    LOG(INFO) << "Got len=" << dataLen << " eof=" << uint32_t(eof)
              << " total=" << input_[id].first.chainLength() + dataLen
              << " data=" << data->clone()->moveToFbString().toStdString();
    input_[id].first.append(std::move(data));
    input_[id].second = eof;
    if (eof) {
      response(id, input_[id]);
    }
  }

  void readError(
      quic::StreamId id,
      std::pair<quic::QuicErrorCode, folly::Optional<folly::StringPiece>>
          error) noexcept override {
    LOG(ERROR) << "Got read error on stream=" << id
               << " error=" << toString(error);
    // A read error only terminates the ingress portion of the stream state.
    // Your application should probably terminate the egress portion via
    // resetStream
  }

  void response(quic::StreamId id, StreamData& data) {
    auto responseData = respBuf_->clone();
    bool eof = false;
    auto res =
        sock->writeChain(id, std::move(responseData), eof, false, nullptr);
    if (res.hasError()) {
      LOG(ERROR) << "write error=" << toString(res.error());
    } else {
      sock->notifyPendingWriteOnStream(id, this);
    }
  }

  void onStreamWriteReady(quic::StreamId id,
                          uint64_t maxToSend) noexcept override {
    VLOG(1) << "socket is write ready with maxToSend=" << maxToSend;
    response(id, input_[id]);
  }

  void onStreamWriteError(
      quic::StreamId id,
      std::pair<quic::QuicErrorCode, folly::Optional<folly::StringPiece>>
          error) noexcept override {
    LOG(ERROR) << "write error with stream=" << id
               << " error=" << toString(error);
  }

  folly::EventBase* getEventBase() { return evb; }

  folly::EventBase* evb;
  std::shared_ptr<quic::QuicSocket> sock;

 private:
  std::map<quic::StreamId, StreamData> input_;
  std::unique_ptr<folly::IOBuf> respBuf_;
};

}  // namespace traffic_gen
}  // namespace quic
