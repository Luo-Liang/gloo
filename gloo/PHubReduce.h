/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstring>
#include <vector>

#include "gloo/algorithm.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"

namespace gloo {

template <typename T>
class PHubReduce : public Algorithm {
 public:
  PHubReduce(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      int count,
      int rootRank = 0,
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum)
      : Algorithm(context),
        ptrs_(ptrs),
        count_(count),
        bytes_(count * sizeof(T)),
        rootRank_(rootRank),
        fn_(fn) {
    GLOO_ENFORCE_GE(rootRank_, 0);
    GLOO_ENFORCE_LT(rootRank_, contextSize_);

    // Setup pairs/buffers for sender/receivers
    if (contextSize_ > 1) {
      auto slot = context_->nextSlot();
      if (contextRank_ == rootRank_) {
        receiver_.resize(contextSize_);
        actualRecvBuffer.resize(contextSize_);
        for (auto i = 0; i < contextSize_; i++) 
        {
          if (i == contextRank_) {
            continue;
          }
          auto ptr = static_cast<T*>(malloc(bytes_));
          actualRecvBuffer[i] = ptr;
          receiver_[i] = make_unique<forReceiver>();
          auto& pair = context_->getPair(i);
          receiver_[i]->clearToSendBuffer = pair->createSendBuffer(
              slot, &receiver_[i]->dummy, sizeof(receiver_[i]->dummy));
          receiver_[i]->recvBuffer = pair->createRecvBuffer(slot, ptr, bytes_);
        }
      } else {
        auto ptr = ptrs[0];
        sender_ = make_unique<forSender>();
        auto& rootPair = context_->getPair(rootRank_);
        sender_->clearToSendBuffer = rootPair->createRecvBuffer(
            slot, &sender_->dummy, sizeof(sender_->dummy));
        sender_->sendBuffer = rootPair->createSendBuffer(slot, ptr, bytes_);
      }
    }
  }

  void run() {
    	for (int i = 1; i < ptrs_.size(); i++) {
				fn_->call(ptrs_[0], ptrs_[i], count_);
			}
			if (this->contextSize_ == 1) {
				// Broadcast ptrs_[0]
				for (int i = 1; i < ptrs_.size(); i++) {
					memcpy(ptrs_[i], ptrs_[0], bytes_);
				}
        //done. i am the only one.
				return;
			}

    if (contextRank_ == rootRank_) 
    {
      // Fire off send operations after receiving clear to send
      //for (auto i = 0; i < contextSize_; i++) {
      //  if (i == contextRank_) {
      //    continue;
      //  }
      //  receiver_[i]->clearToSendBuffer->send();
      //receiver_[i]->sendBuffer->send();
      //}
      // Wait for all recv operations to complete
      //std::vector<bool> recvedFlags(contextSize_);
      int recvedCnt = 1;
      while(recvedCnt < contextSize_)
      {
        for (auto i = 0; i < contextSize_; i++) {
          if (i == contextRank_) {
            continue;
          }
          if(receiver_[i]->recvBuffer->hasRecved())
          {
            receiver_[i]->recvBuffer->waitRecv();
            recvedCnt++;
            fn_->call(ptrs_[0], actualRecvBuffer[i], count_);
          }
        }
      }
      //dont do anything.
      //broadcastLocally();
    } 
    else {
      //sender_->clearToSendBuffer->waitRecv();
      sender_->sendBuffer->send();
      sender_->sendBuffer->waitSend();
      //didnt do anything.
      // Broadcast locally after receiving from root
      //broadcastLocally();
    }
  }
 ~PHubReduce()
 {
   for(T* ptr : actualRecvBuffer)
   {
     if(ptr!=nullptr)
     {
      free(ptr);
     }
   }
 }
 protected:

  // Broadcast from root pointer to other pointers
  void broadcastLocally() {
    for (auto i = 0; i < ptrs_.size(); i++) {
      if (i == rootPointerRank_) {
        continue;
      }

      memcpy(ptrs_[i], ptrs_[rootPointerRank_], bytes_);
    }
  }

  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const int rootRank_;
  const int rootPointerRank_ = 0;
	const ReductionFunction<T>* fn_;

  // For the sender (root)
  using forSender = struct {
    int dummy;
    std::unique_ptr<transport::Buffer> clearToSendBuffer;
    std::unique_ptr<transport::Buffer> sendBuffer;
  };

  std::unique_ptr<forSender> sender_;

  // For all receivers
  using forReceiver = struct {
    int dummy;
    std::unique_ptr<transport::Buffer> clearToSendBuffer;
    std::unique_ptr<transport::Buffer> recvBuffer;
  };
  std::vector<T*> actualRecvBuffer;
  std::vector<std::unique_ptr<forReceiver>> receiver_;
};

} // namespace gloo
