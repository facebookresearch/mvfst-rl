#include "CongestionControlRPCEnv.h"

using namespace grpc;
using namespace rpcenv;

namespace quic {

grpc::Status CongestionControlRPCEnv::StreamingEnv(
    ServerContext* context,
    ServerReaderWriter<rpcenv::Step, rpcenv::Action>* stream) {
  LOG(INFO) << "StreamingEnv initiated";

  // TODO (viswanath): impl
  return grpc::Status::OK;
}

}  // namespace quic
