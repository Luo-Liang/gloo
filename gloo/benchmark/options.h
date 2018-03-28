/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <string>
#include <vector>

#include "gloo/config.h"

namespace gloo {
namespace benchmark {

struct options {
  int contextRank = 0;
  int contextSize = 0;

  // Rendezvous using Redis
  std::string redisHost;
  int redisPort = 6379;
  std::string prefix = "prefix";

  // Rendezvous using MPI
#if GLOO_USE_MPI
  bool mpi = false;
#endif

  // Rendezvous using file system
  std::string sharedPath;
  std::string plinkScheduleFile;
  // Transport
  std::string transport;
  std::vector<std::string> tcpDevice;
  std::vector<std::string> ibverbsDevice;
  int ibverbsPort = 1;
  int ibverbsIndex = 0;
  bool sync = false;
  bool busyPoll = false;

  // Suite configuration
  std::string benchmark;
  bool verify = false;
  int elements = -1;
  long iterationCount = -1;
  long iterationTimeNanos = 2 * 1000 * 1000 * 1000;
  int warmupIterationCount = 5;
  bool showNanos = false;
  int inputs = 1;
  bool gpuDirect = false;
  bool halfPrecision = false;
  int destinations  = 1;
  int threads = 1;
  int base = 2;
};

struct options parseOptions(int argc, char** argv);

} // namespace benchmark
} // namespace gloo
