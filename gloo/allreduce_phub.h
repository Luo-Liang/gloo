#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/common/error.h"
#include "gloo/context.h"
#include <PHub.h>

namespace gloo
{
std::string pHubGetOptionalEnvironmentVariable(std::string name)
{
    var val = std::getenv(name.c_str());
    var ret = val == NULL ? std::string("") : std::string(val);
    return ret;
}

std::string pHubGetMandatoryEnvironmemtVariable(std::string name)
{
    var val = pHubGetOptionalEnvironmentVariable(name);
    CHECK(val != "") << name << " is not set in environment variable";
    return val;
}

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
        var redisHost = pHubGetMandatoryEnvironmemtVariable("PHubRedisHost");
        std::string redisIp;
        uint redisPort;
        ParseHostPort(redisHost, redisIp, redisPort);
        //PHub requests a node IP map. Gloo does this automatically,
        //so we need to use PhubRendezvous to retrieve one.
        std::vector<NodeId> PHubNodes;
        for (int i = 0; i < context->size; i++)
        {
            PHubNodes.push_back((NodeId)i);
        }
        PHubRendezvous rendezvous(redisIp, redisPort, context->rank);
        rendezvous.Connect();
        std::string myIp = trim(PHubExecute("hostname -i"));
        var nodeMap = rendezvous.PullNodeMap(PHubNodes, myIp);
        //remember to chunk my keys

        var chunkSize = atoi(pHubGetMandatoryEnvironmemtVariable("PHubChunkElementSize").c_str());

        int keyCount = (int)ceil(count * 1.0 / chunkSize);
        int chunksSizeInBytes = chunkSize * sizeof(T);
        int remainderInBytes = (count % chunkSize) * sizeof(T);
        CHECK(sizeof(T) == 4) << "Currently PHub only supports 4 Byte floats";
        std::vector<float> keySizes;
        std::vector<void *> appAddrs;
        std::vector<PLinkKey> keys;
        T *ptr = ptrs.at(0);

        //now initialize keySizes.
        T *currPtr = ptr;
        for (PLinkKey key = 0; key < keyCount; key++)
        {
            int sizeInBytes = (key == keyCount - 1) && remainderInBytes != 0 ? remainderInBytes : chunksSizeInBytes;
            int elementCount = sizeInBytes / sizeof(T);
            keySizes.push_back(sizeInBytes);
            appAddrs.push_back(currPtr);
            keys.push_back(key);
            currPtr += elementCount;
        }
        CHECK(currPtr == ptr + count) << "There is a problem assigning appAddr for PHub";

        PHubLaunchPreference preference;
#pragma region
        preference.BlockedInterfaces = pHubGetOptionalEnvironmentVariable("PHubBlockedInterfaces");
        if (pHubGetOptionalEnvironmentVariable("PHubCoreOffset") != "")
        {
            preference.CoreMapLaunchOffset = atoi(pHubGetOptionalEnvironmentVariable("PHubCoreOffset").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubCQDepth") != "")
        {
            preference.CQDepth = atoi(pHubGetOptionalEnvironmentVariable("PHubCQDepth").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubCQSendPollFrequency") != "")
        {
            preference.CQSendPollFrequency = atoi(pHubGetOptionalEnvironmentVariable("PHubCQSendPollFrequency").c_str());
        }
        preference.DisableHyperThreadingWarning = pHubGetOptionalEnvironmentVariable("PHubDisableHyperThreadingWarning") != "";
        preference.DisableNumaAwareness = pHubGetOptionalEnvironmentVariable("PHubDisableNumaAwareness") != "";
        if (pHubGetOptionalEnvironmentVariable("PHubMaximumCore") != "")
        {
            preference.MaximumCores = atoi(pHubGetOptionalEnvironmentVariable("PHubMaximumCore").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubMinRNRTimer") != "")
        {
            preference.MinRNRTimer = atoi(pHubGetOptionalEnvironmentVariable("PHubMinRNRTimer").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubCoreAssignmentScheme") != "")
        {
            preference.PHubCoreAssignmentScheme = pHubGetOptionalEnvironmentVariable("PHubCoreAssignmentScheme");
        }
        if (pHubGetOptionalEnvironmentVariable("PHubIFAssignmentScheme") != "")
        {
            preference.PHubIFAssignmentScheme = pHubGetOptionalEnvironmentVariable("PHubIFAssignmentScheme");
        }
        if (pHubGetOptionalEnvironmentVariable("PHubPreferredGidIndex") != "")
        {
            preference.PreferredGidIndex = atoi(pHubGetOptionalEnvironmentVariable("PHubPreferredGidIndex").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubQPPerDeviceConnection") != "")
        {
            preference.QPPerDeviceConnection = atoi(pHubGetOptionalEnvironmentVariable("PHubQPPerDeviceConnection").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubRetryCount") != "")
        {
            preference.RetryCount = atoi(pHubGetOptionalEnvironmentVariable("PHubRetryCount").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubRNRRetry") != "")
        {
            preference.RNRRetry = atoi(pHubGetOptionalEnvironmentVariable("PHubRNRRetry").c_str());
        }
        if (pHubGetOptionalEnvironmentVariable("PHubTransmitTimeout") != "")
        {
            preference.TransmitTimeout = atoi(pHubGetOptionalEnvironmentVariable("PHubTransmitTimeout").c_str());
        }
        preference.UseiWarp = pHubGetOptionalEnvironmentVariable("PHubUseiWarp") != "False";

#pragma endregion
        //PHub pub(redisHost, nodeMap, keySizes, appAddrs, (int)PHubNodes.size(), sizeof(T), context->rank, preference);
        pHub = std::make_shared<PHub>(redisHost, nodeMap, keySizes, appAddrs, (int)PHubNodes.size(), sizeof(T), context->rank, preference);
        pHub->Initialize();
        //PHubSchedule(std::string scheduleFile, NodeId myId, std::vector<PLinkKey> & keys);
        var scheduleFile = pHubGetMandatoryEnvironmemtVariable("PHubScheduleFile");
        auto pSchedule = std::make_shared<PHubSchedule>(scheduleFile, context->rank, keys);
        pHub->InitializePHubThreading(pSchedule);
    }

    void run()
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