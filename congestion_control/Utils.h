/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <folly/Format.h>
#include <torch/torch.h>

namespace quic {
namespace utils {

// Redefitions of torch::aten_to_numpy_dtype and torch::numpy_dtype_to_aten
// with hardcoded values for NPY_* macros which we can't use since we don't have
// a Python interpreter.

// Ref NPY_TYPES enum in
// https://github.com/numpy/numpy/blob/464f79eb1d05bf938d16b49da1c39a4e02506fa3/numpy/core/include/numpy/ndarraytypes.h.
enum NPY_TYPES {
  NPY_BOOL = 0,
  NPY_BYTE,
  NPY_UBYTE,
  NPY_SHORT,
  NPY_USHORT,
  NPY_INT,
  NPY_UINT,
  NPY_LONG,
  NPY_ULONG,
  NPY_LONGLONG,
  NPY_ULONGLONG,
  NPY_FLOAT,
  NPY_DOUBLE,
  NPY_LONGDOUBLE,
  NPY_CFLOAT,
  NPY_CDOUBLE,
  NPY_CLONGDOUBLE,
  NPY_OBJECT = 17,
  NPY_STRING,
  NPY_UNICODE,
  NPY_VOID,
  /*
   * New 1.6 types appended, may be integrated
   * into the above in 2.0.
   */
  NPY_DATETIME,
  NPY_TIMEDELTA,
  NPY_HALF,

  NPY_NTYPES,
  NPY_NOTYPE,
  NPY_CHAR,
  NPY_USERDEF = 256, /* leave room for characters */

  /* The number of types not including the new 1.6 types */
  NPY_NTYPES_ABI_COMPATIBLE = 21
};

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
}  // namespace quic::utils
