/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <variant>

namespace nest {
// Magic from https://en.cppreference.com/w/cpp/utility/variant/visit
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...)->overloaded<Ts...>;

template <typename T>
struct Nest {
  using value_t =
      std::variant<T, std::vector<Nest>, std::map<std::string, Nest>>;
  Nest(std::vector<T> entries) : value(std::vector<Nest>()) {
    auto &v = std::get<std::vector<Nest>>(value);
    v.reserve(entries.size());
    for (auto &e : entries) {
      v.emplace_back(e);
    }
  }
  Nest(std::map<std::string, T> entries)
      : value(std::map<std::string, Nest>()) {
    auto &m = std::get<std::map<std::string, Nest>>(value);
    for (auto &p : entries) {
      m.emplace_hint(m.end(), p.first, p.second);
    }
  }
  Nest() = default;  // needed for type_caster below.
  Nest(const Nest &) = default;
  Nest(Nest &&) = default;
  Nest &operator=(const Nest &) = default;

  Nest(value_t v) : value(std::move(v)) {}

  value_t value;

  bool is_vector() {
    return std::holds_alternative<std::vector<Nest<T>>>(value);
  }

  std::vector<Nest<T>> &get_vector() {
    return std::get<std::vector<Nest<T>>>(value);
  }

  const std::vector<Nest<T>> &get_vector() const {
    return std::get<std::vector<Nest<T>>>(value);
  }

  T &front() {
    return std::visit(overloaded{[](T &t) -> T & { return t; },
                                 [](std::vector<Nest> &v) -> T & {
                                   return v.front().front();
                                 },
                                 [](std::map<std::string, Nest> &m) -> T & {
                                   return m.begin()->second.front();
                                 }},
                      value);
  }

  const T &front() const {
    return std::visit(
        overloaded{[](const T &t) -> const T & { return t; },
                   [](const std::vector<Nest> &v) -> const T & {
                     return v.front().front();
                   },
                   [](const std::map<std::string, Nest> &m) -> const T & {
                     return m.cbegin()->second.front();
                   }},
        value);
  }

  bool empty() const {
    return std::visit(
        overloaded{[](const T &t) { return false; },
                   [](const std::vector<Nest> &v) {
                     return std::all_of(v.begin(), v.end(),
                                        [](auto &n) { return n.empty(); });
                   },
                   [](const std::map<std::string, Nest> &m) {
                     return std::all_of(m.begin(), m.end(), [](auto &p) {
                       return p.second.empty();
                     });
                   }},
        value);
  }

  template <typename Function>
  Nest<std::invoke_result_t<Function, T>> map(Function f) const {
    using S = std::invoke_result_t<Function, T>;
    return std::visit(overloaded{[&f](const T &t) { return Nest(f(t)); },
                                 [&f](const std::vector<Nest> &v) {
                                   std::vector<Nest<S>> result;
                                   result.reserve(v.size());
                                   for (const Nest<S> &n : v) {
                                     result.emplace_back(n.map(f));
                                   }
                                   return Nest<S>(result);
                                 },
                                 [&f](const std::map<std::string, Nest> &m) {
                                   std::map<std::string, Nest<S>> result;
                                   for (const auto &p : m) {
                                     result.emplace_hint(result.end(), p.first,
                                                         p.second.map(f));
                                   }
                                   return Nest<S>(result);
                                 }},
                      value);
  }

  std::vector<T> flatten() const {
    std::vector<T> result;
    flatten(std::back_inserter(result));
    return result;
  }

  template <class OutputIt>
  OutputIt flatten(OutputIt first) const {
    std::visit(overloaded{
                   [&first](const T &t) { *first++ = t; },
                   [&first](const std::vector<Nest> &v) {
                     for (const Nest &n : v) {
                       n.flatten(first);
                     }
                   },
                   [&first](const std::map<std::string, Nest> &m) {
                     for (auto &p : m) {
                       p.second.flatten(first);
                     }
                   },
               },
               value);
    return first;
  }

  template <class InputIt>
  Nest pack_as(InputIt first, InputIt last) const {
    Nest result = pack_as(&first, last);
    if (first != last) {
      throw std::range_error("Nest didn't exhaust sequence");
    }
    return result;
  }

  template <class InputIt>
  Nest pack_as(InputIt *first, const InputIt &last) const {
    return std::visit(
        overloaded{[&first, &last](const T &) {
                     if (*first == last)
                       throw std::out_of_range("Too few elements in sequence");
                     return Nest(*(*first)++);
                   },
                   [&first, &last](const std::vector<Nest> &v) {
                     std::vector<Nest> result;
                     result.reserve(v.size());
                     for (const Nest &n : v) {
                       result.emplace_back(n.pack_as(first, last));
                     }
                     return Nest(result);
                   },
                   [&first, &last](const std::map<std::string, Nest> &m) {
                     std::map<std::string, Nest> result;
                     for (auto &p : m) {
                       result.emplace_hint(result.end(), p.first,
                                           p.second.pack_as(first, last));
                     }
                     return Nest(result);
                   }},
        value);
  }

  template <typename Function>
  static Nest<std::invoke_result_t<Function, std::vector<T>>> map(
      Function f, const std::vector<Nest<T>> &nests) {
    // Inefficient implementation.
    std::vector<std::vector<T>> flattened;
    for (const Nest<T> &n : nests) {
      flattened.emplace_back();
      n.flatten(std::back_inserter(flattened.back()));
    }

    std::vector<std::vector<T>> transp(flattened[0].size(), std::vector<T>());

    for (unsigned int i = 0; i < flattened.size(); i++) {
      for (unsigned int j = 0; j < flattened[i].size(); j++) {
        transp[j].push_back(flattened[i][j]);
      }
    }

    const Nest<T> &first_nest = *nests.begin();

    int i = 0;

    return first_nest.map([&i, &f, &transp](const T &) {
      int j = i++;
      return f({transp[j].begin(), transp[j].end()});
    });
  }

  /*
  template <typename S>
  static Nest<S> map(const std::function<S(const std::vector<T> &)> &f,
                     const std::vector<Nest<T>> &nests) {
    if (nests.empty())
      throw std::invalid_argument("Cannot map an empty list of nests");

    return std::visit(
        overloaded{
            [&f, &nests](const T &t) {
              std::vector<const T &> args;
              for (const auto &n : nests) {
                args.push_back(std::get<T>(n));
              }
              return Nest<S>(f(args));
            },
            [&f, &nests](const std::vector<Nest> &v) {
              auto size = v.size();
              std::vector<Nest<S>> result;
              result.reserve(size);

              std::vector<const std::vector<Nest<T>> &> vectors;
              for (const auto &n : nests) {
                vectors.push_back(std::get<std::vector<Nest<T>>>(n));
                if (vectors.back().size() != size) {
                  throw std::invalid_argument("Sizes don't match");
                }
              }
              std::vector<const T &> args(vectors.size());

              for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < args.size(); ++j) {
                  args[j] = vectors[j][i];
                }
                result.emplace_back(map(f, args));
              }
              return Nest<S>(result);
            },
            [&f, &nests](std::map<std::string, Nest> &m) {
              // continue here
            },
        },
        nests.front().value);
  }*/

  template <typename Function, typename T1, typename T2>
  static Nest<std::invoke_result_t<Function, T1, T2>> map2(
      Function f, const Nest<T1> nest1, const Nest<T2> nest2) {
    using S = std::invoke_result_t<Function, T1, T2>;
    return std::visit(
        overloaded{
            [&f](const T1 &t1, const T2 &t2) { return Nest<S>(f(t1, t2)); },
            [&f](const std::vector<Nest<T1>> &v1,
                 const std::vector<Nest<T2>> &v2) {
              auto size = v1.size();
              if (size != v2.size()) {
                throw std::invalid_argument(
                    "Expected vectors of same length but got " +
                    std::to_string(size) + " vs " + std::to_string(v2.size()));
              }
              std::vector<Nest<S>> result;
              result.reserve(size);
              for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end();
                   ++it1, ++it2) {
                result.emplace_back(map2(f, *it1, *it2));
              }
              return Nest<S>(result);
            },
            [&f](const std::map<std::string, Nest> &m1,
                 const std::map<std::string, Nest> &m2) {
              auto size = m1.size();
              if (size != m2.size()) {
                throw std::invalid_argument(
                    "Expected maps of same length but got " +
                    std::to_string(size) + " vs " + std::to_string(m2.size()));
              }
              std::map<std::string, Nest<S>> result;
              for (auto it1 = m1.begin(), it2 = m2.begin(); it1 != m1.end();
                   ++it1, ++it2) {
                if ((*it1).first != (*it2).first) {
                  throw std::invalid_argument(
                      "Expected maps to have same keys but found '" +
                      (*it1).first + "' vs '" + (*it2).first + "'");
                }
                result.emplace_hint(result.end(), (*it1).first,
                                    map2(f, (*it1).second, (*it2).second));
              }
              return Nest<S>(result);
            },
            [](auto &&arg1, auto &&arg2) -> Nest<S> {
              throw std::invalid_argument("nests don't match");
            }},
        nest1.value, nest2.value);
  }

  template <class Function>
  Function for_each(Function f) {
    std::visit(overloaded{f,
                          [&f](std::vector<Nest> &v) {
                            for (Nest &n : v) {
                              n.for_each(f);
                            }
                          },
                          [&f](std::map<std::string, Nest> &m) {
                            for (auto &p : m) {
                              p.second.for_each(f);
                            }
                          }},
               value);

    return std::move(f);
  }
};
}  // namespace nest
