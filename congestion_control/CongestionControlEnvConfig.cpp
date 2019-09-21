/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include "CongestionControlEnvConfig.h"

#include <folly/String.h>
#include <glog/logging.h>

namespace quic {

void CongestionControlEnvConfig::parseActionsFromString(
    const std::string& actionsStr) {
  CHECK(!actionsStr.empty()) << "Actions cannot be empty.";

  std::vector<folly::StringPiece> v;
  folly::split(",", actionsStr, v);
  std::vector<std::pair<ActionOp, float>> actions(v.size());

  CHECK_EQ(v[0], "0") << "First action must be no-op (\"0\"), received "
                      << actionsStr;
  actions[0] = {ActionOp::NOOP, 0};

  for (size_t i = 1; i < v.size(); ++i) {
    CHECK_GT(v[i].size(), 1) << "Invalid actions specified: " << actionsStr;
    const char op = v[i][0];
    const auto& val = v[i].subpiece(1);
    actions[i] = {charToActionOp(op), folly::to<float>(val)};
  }

  this->actions = actions;
}

CongestionControlEnvConfig::ActionOp CongestionControlEnvConfig::charToActionOp(
    const char op) {
  switch (op) {
    case '0':
      return ActionOp::NOOP;
    case '+':
      return ActionOp::ADD;
    case '-':
      return ActionOp::SUB;
    case '*':
      return ActionOp::MUL;
    case '/':
      return ActionOp::DIV;
    default:
      LOG(FATAL) << "Unknown char for ActionOp: " << op;
  }
  __builtin_unreachable();
}

}  // namespace quic
