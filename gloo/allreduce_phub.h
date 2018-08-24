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
std::string pHubGetEnvironmemtVariable(std::string &name)
{
    var val = std::getenv(name.c_str());
    CHECK(val != NULL) << name << " is not set in environment variable";
    return std::string(val);
}

template <typename T>
class AllReducePHub : public Algorithm
{
  private:
    std::shared_ptr<PHub> backend;

  public:
    AllReducePHub(
        const std::shared_ptr<::gloo::Context> &context,
        const std::vector<T *> ptrs,
        const int count,
        const ReductionFunction<T> *fn = ReductionFunction<T>::sum) : Algorithm(context)
    {
        //context is not used.
        //this is because PHub uses a separate way of performing rendezvous.
        //but size and rank is still used.
        var redisHost = pHubGetEnvironmemtVariable("PHubRedisHost");
        std::string redisIp;
        ushort redisPort;
        ParseHostPort(redisHost, redisIp, redisPort);
        //PHub requests a node IP map. Gloo does this automatically, 
        //so we need to use PhubRendezvous to retrieve one.
        std::vector<NodeIp> participants;
        for(int i = 0; i < context->size; i++)
        {
            participants.push_back((NodeId)i);
        }
        PHubRendezvous rendezvous(redisIp, redisPort, context->rank);
        std::string myIp = trim(PHubExecute("hostname -i"));
        var nodeMap =rendezvous->PullNodeMap(participants, myIp);
        //remember to chunk my keys
        
        var chunkSize = atoi(pHubGetEnvironmemtVariable("PHubChunkElementSize").c_str());
        T* ptr = ptrs.at(0);
        
        int chunkCount = (int)ceil(count * 1.0 / chunkSize);
        int chunksSizeInBytes = chunkSize * sizeof(T);
        CHECK(sizeof(T) == 4) << "Currently PHub only supports 4 Byte floats";
        std::vector<float> keySizes;
        std::vector<void*> appAddrs;
        PHubLaunchPreference preference;
        backend = std::make_shared<PHub>(redisHost, nodeMap, keySizes, appAddrs, participants.size(), sizeof(T), context->rank, preference);
    }
};
} // namespace gloo