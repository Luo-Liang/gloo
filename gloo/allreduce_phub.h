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
    std::vector<PLinkKey> reductionKeys;

  public:
    bool UseStandAlonePHub;
    AllReducePHub(
        const std::shared_ptr<::gloo::Context> &context,
        const std::vector<T *> ptrs,
        const int count,
        const ReductionFunction<T> *fn = ReductionFunction<T>::sum,
        bool useStandAlonePHub = true) : Algorithm(context),
                                         dataElementCount(count),
                                         ptrs_(ptrs),
                                         fn_(fn),
                                         UseStandAlonePHub(useStandAlonePHub)
    {
        //context is not used.
        //this is because PHub uses a separate way of performing rendezvous.
        //but size and rank is still used.
        if (context->size == 1)
            return;

        std::string standAlone = pHubGetOptionalEnvironmentVariable("PHubStandAlone");
        UseStandAlonePHub = standAlone != "False";
        if (UseStandAlonePHub)
        {
            pHub = createPHubInstance(ptrs_.at(0), count, context_->size, context_->rank, ::gloo::Context::getCID());
            reductionKeys = pHub->inferredKeys;
            //printf("[%d] Creating PHub Instance.\r\n", context_->rank);
        }

        //pHub = getPHubInstance(ptrs_.at(0), count, context->size, context->rank, ::gloo::Context::getCID());
        //printf("[%d] PHub initialized at %p\n", context->rank, this);
    }
    ~AllReducePHub()
    {
        pHub->Stop();
        //pHub->ShowPerformanceStatistics();
    }

    void runSharedPHubInitialization(std::string frameworkSpecifics)
    {
      #ifndef PHUB_CHECK
      #define PHUB_CHECK CHECK
      #endif
      PHUB_CHECK(UseStandAlonePHub == false);
        caffe2BuildPHubInstance(
            frameworkSpecifics,
            (float *)ptrs_[0],
            dataElementCount,
            context_->size,
            context_->rank);
        reductionKeys = caffe2KeyGetPLinkKey(ptrs_[0]);
        pHub = getPHubInstance();
    }

    virtual void run()
    {
        for (int i = 1; i < ptrs_.size(); i++)
        {
            fn_->call(ptrs_[0], ptrs_[i], dataElementCount);
        }
        if (this->contextSize_ != 1)
        {
        //simply call PHub Reduce.
        //CHECK(pHub != NULL || UseStandAlonePHub == false);
	  pHub->Reduce(reductionKeys);
	}
	for (int i = 1; i < ptrs_.size(); i++)
	  {
	    memcpy(ptrs_[i], ptrs_[0], sizeof(T) * dataElementCount);
	  }

    }
};
} // namespace gloo
