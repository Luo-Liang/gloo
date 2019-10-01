#include "gloo/cuda_allreduce_phub.h"

#include "gloo/cuda_collectives_device.h"
#include "gloo/cuda_collectives_host.h"
#include "gloo/cuda_private.h"

namespace gloo
{

template <typename T, typename W>
CudaAllreducePHub<T, W>::CudaAllreducePHub(
    const std::shared_ptr<Context> &context,
    const std::vector<T *> &ptrs,
    const int count,
    const std::vector<cudaStream_t> &streams)
    : Algorithm(context),
      count_(count),
      bytes_(count_ * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      fn_(CudaReductionFunction<T>::sum)
{
  auto newStream = true;
  if (streams.size() > 0)
  {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (auto i = 0; i < ptrs.size(); i++)
  {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    if (newStream)
    {
      streams_.push_back(CudaStream(ptr.getDeviceID()));
    }
    else
    {
      streams_.push_back(CudaStream(ptr.getDeviceID(), streams[i]));
    }
    devicePtrs_.push_back(std::move(ptr));
  }

  // Workspace specific initialization (see below)
  init();

  // inbox_ is ready.

  if (this->context_->size == 1)
  {
    return;
  }
  T *ptr = *inbox_;
  int cid = ::gloo::Context::getCID();

  std::string standAlone = pHubGetOptionalEnvironmentVariable("PHubStandAlone");
  UseStandAlonePHub = standAlone != "False";
  if (UseStandAlonePHub)
  {
    pHub = createPHubInstance(ptr, count, context->size, context->rank, cid);
    reductionKeys = pHub->inferredKeys;
  }
}

template <typename T, typename W>
void CudaAllreducePHub<T, W>::runSharedPHubInitialization(std::string frameworkSpecifics)
{
  CHECK(UseStandAlonePHub == false);
  CHECK(count_ > 0 && bytes_ > 0);
  caffe2BuildPHubInstance(
      frameworkSpecifics,
      (float*)*inbox_,
      count_,
      context_->size,
      context_->rank);
  reductionKeys = caffe2KeyGetPLinkKey(*inbox_);
  pHub = getPHubInstance();
}

// namespace gloo
template <typename T, typename W>
void CudaAllreducePHub<T, W>::run()
{
  CudaDeviceGuard guard;
  CudaStream &stream = *scratchStream_;

  if (localReduceOp_)
  {
    localReduceOp_->run();
  }

  // Initialize outbox with locally reduced values
  if (this->context_->size != 1)
  {
    stream.copyAsync(inbox_, scratch_);
    stream.wait();
    var reductionBuffer = (float*)(*inbox_);
    for(int i = 0; i < count_; i++)
    {
        reductionBuffer[i] = 1;
    }
    pHub->Reduce(reductionKeys);
    for(int i = 0; i < count_; i++)
    {
        GLOO_ENFORCE(reductionBuffer[i] == context_->size); // << " location = " << i << " " << reductionBuffer[i] << " vs " <<context_->size;
        reductionBuffer[i] = 1;
    }
    fprintf(stderr, "information: [%d] [%p] verified\n", context_->rank, *inbox_);
    stream.copyAsync(scratch_, inbox_);
    stream.wait();
  }

  if (localBroadcastOp_)
  {
    localBroadcastOp_->runAsync();
    if (synchronizeDeviceOutputs_)
    {
      localBroadcastOp_->wait();
    }
  }
}

template <typename T, typename W>
template <typename U>
void CudaAllreducePHub<T, W>::init(typename std::enable_if<
                                   std::is_same<U, CudaHostWorkspace<T>>::value,
                                   typename U::Pointer>::type *)
{
  // Since reduction is executed on the CPU, the scratch space
  // where they are accumulated is a new host side buffer.
  scratch_ = W::Pointer::alloc(count_);
  scratchStream_ = &streams_[0];

  // Execute local reduction and broadcast from host.
  // If devicePtrs_.size() == 1 these functions construct an op that
  // executes a memcpy such that scratch_ always holds the result.
  localReduceOp_ =
      cudaHostReduce(streams_, devicePtrs_, scratch_, fn_, 0, count_);
  localBroadcastOp_ =
      cudaHostBroadcast(streams_, devicePtrs_, scratch_, 0, count_);

  inbox_ = W::Pointer::alloc(count_);
}

template <typename T, typename W>
template <typename U>
void CudaAllreducePHub<T, W>::init(
    typename std::enable_if<
        std::is_same<U, CudaDeviceWorkspace<T>>::value,
        typename U::Pointer>::type *)
{
  // The networking adapter does DMA to/from GPU memory, so we should reduce
  // onto the device that's closest to the networking adapter bound
  // to our context. This uses PCI distance to find closest GPU.
  auto index = findCudaDevicePointerClosestToDevice(
      devicePtrs_, this->context_->getDevice());
  scratch_ = CudaDevicePointer<T>::create(devicePtrs_[index]);
  scratchStream_ = &streams_[index];

  // Run local reduction and broadcast on device.
  // When running with a device workspace we intend to never leave the device.
  if (devicePtrs_.size() > 1)
  {
    localReduceOp_ =
        cudaDeviceReduce(streams_, devicePtrs_, scratch_, fn_, 0, count_);
    localBroadcastOp_ =
        cudaDeviceBroadcast(streams_, devicePtrs_, scratch_, 0, count_);
  }

  // Inbox/outbox must be colocated with scratch buffer to avoid
  // cross device copies while accumulating the reduction.
  {
    CudaDeviceScope scope(scratch_.getDeviceID());
    inbox_ = W::Pointer::alloc(count_);
  }
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                              \
  template class CudaAllreducePHub<T, CudaHostWorkspace<T>>; \
  template class CudaAllreducePHub<T, CudaDeviceWorkspace<T>>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(uint8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
