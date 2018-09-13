#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/common/error.h"
#include "gloo/context.h"
#include <PHub/PHub.h>
#include <PHub/Integration.h>

namespace gloo
{
template <typename T>
class AllReducePHub : public Algorithm
{
  private:
    std::shared_ptr<PHub> pHub;
    std::vector<T *> ptrs_;
    int dataElementCount;
    const ReductionFunction<T> *fn_;

  public:
    AllReducePHub(
        const std::shared_ptr<::gloo::Context> &context,
        const std::vector<T *> ptrs,
        const int count,
        const ReductionFunction<T> *fn = ReductionFunction<T>::sum) : Algorithm(context),
                                                                      dataElementCount(count),
                                                                      ptrs_(ptrs),
                                                                      fn_(fn)
    {
        //context is not used.
        //this is because PHub uses a separate way of performing rendezvous.
        //but size and rank is still used.
        if (context->size == 1)
            return;
        pHub = createPHubInstance(ptrs_.at(0), count, context->size, context->rank, context->getCID());
        //printf("[%d] PHub initialized at %p\n", context->rank, this);
    }

    virtual void run()
    {
        for (int i = 1; i < ptrs_.size(); i++)
        {
            fn_->call(ptrs_[0], ptrs_[i], dataElementCount);
        }
        if (this->contextSize_ == 1)
        {
            // Broadcast ptrs_[0]
            for (int i = 1; i < ptrs_.size(); i++)
            {
                memcpy(ptrs_[i], ptrs_[0], sizeof(T) * dataElementCount);
            }
            return;
        }

        //simply call PHub Reduce.
        pHub->Reduce();
    }
};
} // namespace gloo
