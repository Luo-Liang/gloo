/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>
#include <vector>

#include "gloo/transport/device.h"
#include "gloo/transport/pair.h"
#include <atomic>

namespace gloo
{

class Context
{
public:
  Context(int rank, int size, int base = 2);
  virtual ~Context();

  const int rank;
  const int size;
  int base;

  std::shared_ptr<transport::Device> &getDevice();

  std::unique_ptr<transport::Pair> &getPair(int i);

  int nextSlot(int numToSkip = 1);

  static int getCID()
  {
    return CIDTicketer.fetch_add(1, std::memory_order::memory_order_relaxed);
  }

  void closeConnections();

  void setTimeout(std::chrono::milliseconds timeout);

  std::chrono::milliseconds getTimeout() const;

protected:
  static std::atomic<int> CIDTicketer;
  std::shared_ptr<transport::Device> device_;
  std::vector<std::unique_ptr<transport::Pair>> pairs_;
  int slot_;
  std::chrono::milliseconds timeout_;
};

} // namespace gloo
