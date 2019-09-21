/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/

/*
c++ nest_test.cc -lgtest -lgtest_main -std=c++17 -stdlib=libc++
-mmacosx-version-min=10.14 -o nest_test
 */

#include <gtest/gtest.h>

#include "nest/nest.h"

namespace {
using namespace nest;

TEST(NestTest, TestConstructDestroy) { Nest<int> n(3); }

TEST(NestTest, TestEmpty) {
  std::vector<Nest<int>> v;
  std::map<std::string, Nest<int>> m;

  Nest<int> n1(v);

  Nest<int> n2(
      std::vector<Nest<int>>({Nest<int>(v), Nest<int>(v), Nest<int>(m)}));

  Nest<int> n3(42);

  Nest<int> n4(
      std::vector<Nest<int>>({Nest<int>(v), Nest<int>(m), Nest<int>(666)}));

  std::vector<Nest<int>> v1({Nest<int>(32)});
  std::map<std::string, Nest<int>> m1;
  m1["integer"] = Nest<int>(69);

  Nest<int> n5(std::vector<Nest<int>>({Nest<int>(v), Nest<int>(v1)}));
  Nest<int> n6(std::vector<Nest<int>>({Nest<int>(v), Nest<int>(m1)}));

  ASSERT_TRUE(n1.empty());
  ASSERT_TRUE(n2.empty());

  ASSERT_FALSE(n3.empty());
  ASSERT_FALSE(n4.empty());

  ASSERT_FALSE(n5.empty());
  ASSERT_FALSE(n6.empty());
}
}  // namespace
