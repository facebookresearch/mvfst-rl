/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#if defined NDEBUG
#include <vector>
#else
#include <debug/vector>
#endif

#include <torch/torch.h>

namespace quic {
namespace utils {

// Define `quic::utils::vector` to be either the regular `std::vector` or its
// debug version (which includes in particular bound checks), depending on
// whether or not we are running in debug mode.
template <class T>
#if defined NDEBUG
using vector = std::vector<T>;
#else
using vector = __gnu_debug::vector<T>;
#endif

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

int aten_to_numpy_dtype(const at::ScalarType scalar_type);

at::ScalarType numpy_dtype_to_aten(int dtype);
}
} // namespace quic::utils
