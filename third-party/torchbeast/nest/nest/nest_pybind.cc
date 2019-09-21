/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nest.h"
#include "nest_pybind.h"

namespace py = pybind11;

typedef nest::Nest<py::object> PyNest;

class py_list_back_inserter {
 public:
  py_list_back_inserter(py::list &l) : list_(&l) {}
  py_list_back_inserter &operator=(const py::object &value) {
    list_->append(value);
    return *this;
  };
  constexpr py_list_back_inserter &operator*() { return *this; };
  constexpr py_list_back_inserter &operator++() { return *this; }
  constexpr py_list_back_inserter &operator++(int) { return *this; }

 private:
  py::list *list_;
};

PYBIND11_MODULE(nest, m) {
  m.def("map", [](py::function f, const PyNest &n) {
    // This says const py::object, but f can actually modify it!
    std::function<py::object(const py::object &)> cppf =
        [&f](const py::object &arg) { return f(arg); };
    return n.map(cppf);
  });
  m.def("map_many",
        [](const std::function<py::object(const std::vector<py::object> &)> &f,
           py::args args) {
          const std::vector<PyNest> nests = args.cast<std::vector<PyNest>>();
          return PyNest::map(f, nests);
        });
  m.def("map_many2", [](const std::function<py::object(const py::object &,
                                                       const py::object &)> &f,
                        const PyNest &n1, const PyNest &n2) {
    try {
      return PyNest::map2(f, n1, n2);
    } catch (const std::invalid_argument &e) {
      // IDK why I have to do this manually.
      throw py::value_error(e.what());
    }
  });
  m.def("flatten", [](const PyNest &n) {
    py::list result;
    n.flatten(py_list_back_inserter(result));
    return result;
  });
  m.def("pack_as", [](const PyNest &n, const py::sequence &sequence) {
    try {
      return n.pack_as(sequence.begin(), sequence.end());
    } catch (const std::exception &e) {
      // PyTorch pybind11 doesn't seem to translate exceptions?
      throw py::value_error(e.what());
    }
  });
  m.def("front", [](const PyNest &n) { return n.front(); });
}
