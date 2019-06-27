cmake_minimum_required(VERSION 3.10)

project(
  mv-rl-fst
)

find_package(mvfst)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(QUIC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third-party/mvfst)

list(APPEND
  CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${QUIC_ROOT}/cmake
)

list(APPEND
  _QUIC_BASE_COMPILE_OPTIONS
  -std=c++14
  -Wall
  -Wextra
)

list(APPEND
  _QUIC_COMMON_COMPILE_OPTIONS
  ${_QUIC_BASE_COMPILE_OPTIONS}
  # more strict options
  -Werror=sign-compare
  -Werror=bool-compare
  -Woverloaded-virtual
  -Wnon-virtual-dtor
)

# Find GFlags
SET(GFLAG_DEPENDENCIES "")
find_package(gflags CONFIG QUIET)
if (gflags_FOUND)
  message(STATUS "Found gflags from package config")
  if (TARGET gflags-shared)
    list(APPEND GFLAG_DEPENDENCIES gflags-shared)
  elseif (TARGET gflags)
    list(APPEND GFLAG_DEPENDENCIES gflags)
  else()
    message(FATAL_ERROR "Unable to determine the target name for the GFlags package.")
  endif()
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${GFLAGS_LIBRARIES})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${GFLAGS_INCLUDE_DIR})
else()
  find_package(Gflags REQUIRED MODULE)
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${LIBGFLAGS_LIBRARY})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${LIBGFLAGS_INCLUDE_DIR})
endif()

# Find GMock and GTest. Required for linking some TestUtils.
set(REQUIRED_LINK_DIRS "")
if(BUILD_TESTS)
  enable_testing()
  list(APPEND CMAKE_REQUIRED_INCLUDES "${QUIC_ROOT}/_build/build/googletest/src/googletest/googlemock/include")
  list(APPEND CMAKE_REQUIRED_INCLUDES "${QUIC_ROOT}/_build/build/googletest/src/googletest/googletest/include")
  list(APPEND REQUIRED_LINK_DIRS "${QUIC_ROOT}/_build/build/googletest/src/googletest-build/googlemock")
  list(APPEND REQUIRED_LINK_DIRS "${QUIC_ROOT}/_build/build/googletest/src/googletest-build/googlemock/gtest")
  list(APPEND CMAKE_REQUIRED_LIBRARIES "gmock")
  list(APPEND CMAKE_REQUIRED_LIBRARIES "gtest")
endif()

include_directories(
  ${CMAKE_REQUIRED_INCLUDES}
  ${PROJECT_ROOT}
)

link_directories(
  ${REQUIRED_LINK_DIRS}
)

link_libraries(
  ${CMAKE_REQUIRED_LIBRARIES}
)

add_subdirectory(traffic_gen)
add_subdirectory(congestion_control)
