/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <memory>
#include <fstream>
#include <vector>
#include <unistd.h>
#include "gloo/allgather_ring.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_bcube.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/allreduce_phub.h"
#include "gloo/PHubReduce.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/pairwise_exchange.h"
#include "gloo/reduce_scatter.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"
#include "gloo/types.h"
#include "gloo/rendezvous/redis_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/runner.h"
#include "third-party/nlohmann/json/single_include/nlohmann/json.hpp"
#include "gloo/transport/tcp/device.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace
{

template <typename T>
class AllgatherBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;

  public:
    virtual void initialize(size_t elements) override
    {
        auto inPtrs = this->allocate(this->options_.inputs, elements);
        GLOO_ENFORCE_EQ(inPtrs.size(), this->options_.inputs);
        outputs_.resize(this->options_.inputs * this->context_->size * elements);
        this->algorithm_.reset(new AllgatherRing<T>(
            this->context_, this->getInputs(), outputs_.data(), elements));
    }

    virtual void verify() override
    {
        const auto stride = this->context_->size * this->inputs_.size();
        const auto elements = this->inputs_[0].size();
        for (int rank = 0; rank < this->context_->size; rank++)
        {
            auto val = rank * this->inputs_.size();
            for (int elem = 0; elem < elements; elem++)
            {
                T exp(elem * stride + val);
                for (int input = 0; input < this->inputs_.size(); input++)
                {
                    const auto rankOffset = rank * elements * this->inputs_.size();
                    const auto inputOffset = input * elements;
                    GLOO_ENFORCE_EQ(
                        outputs_[rankOffset + inputOffset + elem], exp + T(input),
                        "Mismatch at index: [", rank, ", ", input, ", ", elem, "]");
                }
            }
        }
    }

  protected:
    std::vector<T> outputs_;
};

template <class A, typename T>
class AllreduceBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;

  public:
    virtual void initialize(size_t elements) override
    {
        auto ptrs = this->allocate(this->options_.inputs, elements);
        for (const auto &ptr : ptrs)
        {
            for (size_t i = 0; i < elements; i++)
            {
                fprintf(stderr, "[%d]:input(%p)[%d] = %f. element size = %d. Setting to 0xDEADBEEF\n", this->context_->rank, &ptr[i] ,i, ptr[i], sizeof(ptr[i]));
                ptr[i] = -6259853398707798000;
            }
            printf("\n");
        }
        this->algorithm_.reset(new A(this->context_, ptrs, elements));
    }

    virtual void verify() override
    {
        // Size is the total number of pointers across the context
        const auto size = this->context_->size * this->inputs_.size();
        // Expected is set to the expected value at ptr[0]
        const auto expected = (size * (size - 1)) / 2;
        // The stride between values at subsequent indices is equal to
        // "size", and we have "size" of them. Therefore, after
        // allreduce, the stride between expected values is "size^2".
        const auto stride = size * size;
        for (const auto &input : this->inputs_)
        {
            for (int i = 0; i < input.size(); i++)
            {
                auto offset = i * stride;
                GLOO_ENFORCE_EQ(
                    T(offset + expected), input[i], "Mismatch at index: ", i);
            }
        }
    }
};

template <typename T>
class PLinkScheduleBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;
    using json = nlohmann::json;
    std::vector<std::shared_ptr<transport::Device>> transportDevices_;
    //std::vector<T> outputs_;
    static int InitID;
    //json schedule;
  public:
    virtual void verify() override
    {
        // Size is the total number of pointers across the context
        const auto size = this->context_->size * this->inputs_.size();
        // Expected is set to the expected value at ptr[0]
        const auto expected = (size * (size - 1)) / 2;
        // The stride between values at subsequent indices is equal to
        // "size", and we have "size" of them. Therefore, after
        // allreduce, the stride between expected values is "size^2".
        const auto stride = size * size;
        for (const auto &input : this->inputs_)
        {
            for (int i = 0; i < input.size(); i++)
            {
                auto offset = i * stride;
                GLOO_ENFORCE_EQ(
                    T(offset + expected), input[i], "Mismatch at index: ", i);
            }
        }
    }
    virtual void initialize(size_t elements) override
    {

        if (this->options_.transport == "tcp")
        {
            if (this->options_.tcpDevice.empty())
            {
                transport::tcp::attr attr;
                transportDevices_.push_back(transport::tcp::CreateDevice(attr));
            }
            else
            {
                for (const auto &name : this->options_.tcpDevice)
                {
                    transport::tcp::attr attr;
                    attr.iface = name;
                    transportDevices_.push_back(transport::tcp::CreateDevice(attr));
                }
            }
        }
        else
        {
            GLOO_ENFORCE(false);
        }
        auto ptrs = this->allocate(this->options_.inputs, elements);
        //outputs_.resize(this->options_.inputs * this->context_->size * elements);
        auto filePath = this->options_.plinkScheduleFile;
        std::ifstream i(filePath);
        json schedule = json::parse(i);
        //build addresses.
        //figure out my schedule.
        std::vector<std::shared_ptr<Algorithm>> mySchedule;
        int layer = 0;
        for (json::iterator it = schedule.begin(); it != schedule.end(); it++)
        {
            json layerSchedule = *it;
            for (json::iterator sched = layerSchedule.begin(); sched != layerSchedule.end(); sched++)
            {
                //"participants"
                //"Algorithm"
                json obj = *sched;
                std::vector<int> participants = obj["participants"];
                std::string algorithm = obj["algorithm"];
                std::string groupId = obj["groupId"];
                int RootId = 0;
                if (obj.find("rootId") != obj.end())
                {
                    RootId = obj["rootId"];
                }
                auto fnd = std::find(participants.begin(), participants.end(), this->options_.contextRank);
                if (fnd != participants.end())
                {
                    //printf("plink initialization layer %d starting. rank=%d. gid=%s\n", layer, this->options_.contextRank,groupId.c_str());

                    //needs to translate this context_rank (aka node id) to an actual context in this set of collectives.
                    auto rnk = fnd - participants.begin();
                    //i should queue this task for me.

                    auto pCtx = std::make_shared<::gloo::rendezvous::Context>(rnk, participants.size());
                    //printf("plink initialization layer %d starting. rank=%d. gid=%s, localrank = %d, max=%d\n", layer, this->options_.contextRank,groupId.c_str(), rnk, participants.size());
                    gloo::rendezvous::RedisStore redisStore(this->options_.redisHost, this->options_.redisPort);
                    gloo::rendezvous::PrefixStore prefixStore("run_" + std::to_string(InitID) + "/" + groupId, redisStore);
                    GLOO_ENFORCE(transportDevices_.size() > 0);
                    pCtx->connectFullMesh(prefixStore, transportDevices_.at(0));
                    //create an algorithm.
                    std::shared_ptr<gloo::Algorithm> algo = NULL;
                    //if (algorithm == "allgather_ring") {
                    //	algo = std::make_shared<AllgatherRing<T>>(pCtx, this->getInputs(), outputs_.data(), elements);
                    //}
                    //else
                    if (algorithm == "allreduce_ring")
                    {
                        algo = std::make_shared<AllreduceRing<T>>(pCtx, ptrs, elements);
                    }
                    else if (algorithm == "allreduce_ring_chunked")
                    {
                        algo = std::make_shared<AllreduceRingChunked<T>>(pCtx, ptrs, elements);
                    }
                    else if (algorithm == "allreduce_halving_doubling")
                    {
                        algo = std::make_shared<AllreduceHalvingDoubling<T>>(pCtx, ptrs, elements);
                    }
                    else if (algorithm == "allreduce_bcube")
                    {
                        algo = std::make_shared<AllreduceBcube<T>>(pCtx, ptrs, elements);
                    }
                    else if (algorithm == "broadcast_one_to_all")
                    {
                        int rootRnk = std::find(participants.begin(), participants.end(), RootId) - participants.begin();
                        algo = std::make_shared<BroadcastOneToAll<T>>(pCtx, ptrs, elements, rootRnk); //rootRank is explicitly specified.
                    }
                    else if (algorithm == "phub_reduce")
                    {
                        algo = std::make_shared<PHubReduce<T>>(pCtx, ptrs, elements);
                    }
                    else
                    {
                        GLOO_ENFORCE(false);
                    }
                    //I can be only in one schedule in one layer.
                    //add to my context.
                    //need to initialize context, because the _backing context will not be sufficient for all schedules.

                    mySchedule.push_back(algo);
                    //printf("plink initialization layer %d done. rank=%d. gid = %s\n", layer, this->options_.contextRank, groupId.c_str());
                }
            }
            layer++;
        }
        this->algorithm_ = std::make_unique<MultiphaseAlgorithm>(mySchedule, this->context_);
        InitID++;
        //printf("plink initialization done. rank=%d\n", this->options_.contextRank);
    }
};

template <typename T>
class BarrierAllToAllBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;

  public:
    virtual void initialize(size_t /* unused */) override
    {
        this->algorithm_.reset(new BarrierAllToAll(this->context_));
    }
};

template <typename T>
class BarrierAllToOneBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;

  public:
    virtual void initialize(size_t /* unused */) override
    {
        // This tool measures at rank=0, so use root=1 for the all to one
        // barrier to measure the end-to-end latency (otherwise we might
        // not account for the send-to-root part of the algorithm).
        this->algorithm_.reset(new BarrierAllToOne(this->context_, 1));
    }
};

template <typename T>
class BroadcastOneToAllBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;

  public:
    virtual void initialize(size_t elements) override
    {
        auto ptrs = this->allocate(this->options_.inputs, elements);
        this->algorithm_.reset(
            new BroadcastOneToAll<T>(this->context_, ptrs, elements, rootRank_));
    }

    virtual void verify() override
    {
        const auto stride = this->context_->size * this->inputs_.size();
        for (const auto &input : this->inputs_)
        {
            for (int i = 0; i < input.size(); i++)
            {
                auto offset = i * stride;
                GLOO_ENFORCE_EQ(
                    T(offset + rootRank_), input[i], "Mismatch at index: ", i);
            }
        }
    }

  protected:
    const int rootRank_ = 0;
};

template <typename T>
class PairwiseExchangeBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;

  public:
    virtual void initialize(size_t elements) override
    {
        this->algorithm_.reset(new PairwiseExchange(
            this->context_, elements, this->getOptions().destinations));
    }
};

template <typename T>
class ReduceScatterBenchmark : public Benchmark<T>
{
    using Benchmark<T>::Benchmark;

  public:
    virtual void initialize(size_t elements) override
    {
        auto ptrs = this->allocate(this->options_.inputs, elements);
        int rem = elements;
        int chunkSize =
            (elements + this->context_->size - 1) / this->context_->size;
        for (int i = 0; i < this->context_->size; ++i)
        {
            recvCounts_.push_back(std::min(chunkSize, rem));
            rem = rem > chunkSize ? rem - chunkSize : 0;
        }
        this->algorithm_.reset(
            new ReduceScatterHalvingDoubling<T>(
                this->context_, ptrs, elements, recvCounts_));
    }

    virtual void verify() override
    {
        // Size is the total number of pointers across the context
        const auto size = this->context_->size * this->inputs_.size();
        // Expected is set to the expected value at ptr[0]
        const auto expected = (size * (size - 1)) / 2;
        // The stride between values at subsequent indices is equal to
        // "size", and we have "size" of them. Therefore, after
        // reduce-scatter, the stride between expected values is "size^2".
        const auto stride = size * size;
        for (const auto &input : this->inputs_)
        {
            int numElemsSoFar = 0;
            for (int i = 0; i < this->context_->rank; ++i)
            {
                numElemsSoFar += recvCounts_[i];
            }
            for (int i = 0; i < recvCounts_[this->context_->rank]; ++i)
            {
                auto offset = (numElemsSoFar + i) * stride;
                GLOO_ENFORCE_EQ(
                    T(offset + expected), input[i], "Mismatch at index: ", i);
            }
        }
    }

  protected:
    std::vector<int> recvCounts_;
};

} // namespace

#define RUN_BENCHMARK(T)                                                         \
    Runner::BenchmarkFn<T> fn;                                                   \
    if (x.benchmark == "allgather_ring")                                         \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<AllgatherBenchmark<T>>(context, x);         \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "allreduce_ring")                                    \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<AllreduceBenchmark<AllreduceRing<T>, T>>(   \
                context, x);                                                     \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "allreduce_ring_chunked")                            \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<                                            \
                AllreduceBenchmark<AllreduceRingChunked<T>, T>>(context, x);     \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "allreduce_halving_doubling")                        \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<                                            \
                AllreduceBenchmark<AllreduceHalvingDoubling<T>, T>>(context, x); \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "allreduce_bcube")                                   \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<                                            \
                AllreduceBenchmark<AllreduceBcube<T>, T>>(context, x);           \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "allreduce_phub")                                    \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<                                            \
                AllreduceBenchmark<AllReducePHub<T>, T>>(context, x);            \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "barrier_all_to_all")                                \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<BarrierAllToAllBenchmark<T>>(context, x);   \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "barrier_all_to_one")                                \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<BarrierAllToOneBenchmark<T>>(context, x);   \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "broadcast_one_to_all")                              \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<BroadcastOneToAllBenchmark<T>>(context, x); \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "pairwise_exchange")                                 \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<PairwiseExchangeBenchmark<T>>(context, x);  \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "reduce_scatter")                                    \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<ReduceScatterBenchmark<T>>(context, x);     \
        };                                                                       \
    }                                                                            \
    else if (x.benchmark == "plink")                                             \
    {                                                                            \
        fn = [&](std::shared_ptr<Context> &context) {                            \
            return gloo::make_unique<PLinkScheduleBenchmark<T>>(context, x);     \
        };                                                                       \
    }                                                                            \
    if (!fn)                                                                     \
    {                                                                            \
        GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);                 \
    }                                                                            \
    Runner r(x);                                                                 \
    r.run(fn);
template <typename T>
int PLinkScheduleBenchmark<T>::InitID = 0;
int main(int argc, char **argv)
{
    auto x = benchmark::parseOptions(argc, argv);
    if (x.gdb)
    {
        int rem = 20;
        while (rem-- > 0)
        {
            printf("[%d] is waiting for GDB attach %ds remaining.\n", x.contextRank, x.gdb);
            sleep(1);
        }
    }
    if (x.benchmark == "pairwise_exchange")
    {
        RUN_BENCHMARK(char);
    }
    else if (x.halfPrecision)
    {
        RUN_BENCHMARK(float16);
    }
    else
    {
        RUN_BENCHMARK(float);
    }
    return 0;
}
