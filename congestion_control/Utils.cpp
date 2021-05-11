/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/

#include <folly/Format.h>

#include "Utils.h"

namespace quic {
namespace utils {

int aten_to_numpy_dtype(const at::ScalarType scalar_type) {
  switch (scalar_type) {
  case at::kDouble:
    return NPY_DOUBLE;
  case at::kFloat:
    return NPY_FLOAT;
  case at::kHalf:
    return NPY_HALF;
  case at::kLong:
    return NPY_LONG;
  case at::kInt:
    return NPY_INT;
  case at::kShort:
    return NPY_SHORT;
  case at::kChar:
    return NPY_BYTE;
  case at::kByte:
    return NPY_UBYTE;
  case at::kBool:
    return NPY_BOOL;
  default:
    throw std::runtime_error(
        folly::sformat("Unsupported ScalarType: {}", toString(scalar_type)));
  }
}

at::ScalarType numpy_dtype_to_aten(int dtype) {
  switch (dtype) {
  case NPY_DOUBLE:
    return at::kDouble;
  case NPY_FLOAT:
    return at::kFloat;
  case NPY_HALF:
    return at::kHalf;
  case NPY_LONG:
    return at::kLong;
  case NPY_INT:
    return at::kInt;
  case NPY_SHORT:
    return at::kShort;
  case NPY_BYTE:
    return at::kChar;
  case NPY_UBYTE:
    return at::kByte;
  case NPY_BOOL:
    return at::kBool;
  default:
    throw std::runtime_error(folly::sformat("Unsupported dtype: {}", dtype));
  }
}
}
} // namespace quic::utils
