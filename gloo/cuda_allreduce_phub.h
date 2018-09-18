#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/allreduce_phub.h"
#include "gloo/common/error.h"
#include "gloo/context.h"
#include "gloo/cuda.h"
#include "gloo/cuda_workspace.h"

#include <PHub/PHub.h>

namespace gloo {

template <typename T, typename W = CudaHostWorkspace<T> >
class CudaAllreducePHub : public Algorithm {
 public:
  CudaAllreducePHub(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual ~CudaAllreducePHub() = default;

  virtual void run() override;
  bool UseStandAlonePHub;

 protected:
  // Both workspace types have their own initialization function.
  template <typename U = W>
  void init(
      typename std::enable_if<
          std::is_same<U, CudaHostWorkspace<T>>::value,
          typename U::Pointer>::type* = 0);

  template <typename U = W>
  void init(
      typename std::enable_if<
          std::is_same<U, CudaDeviceWorkspace<T>>::value,
          typename U::Pointer>::type* = 0);

  std::vector<CudaDevicePointer<T>> devicePtrs_;
  std::vector<CudaStream> streams_;
  typename W::Pointer scratch_;
  CudaStream* scratchStream_;

  const int count_;
  const int bytes_;
  const bool synchronizeDeviceOutputs_;
  const CudaReductionFunction<T>* fn_;

  std::unique_ptr<LocalOp<T>> localReduceOp_;
  std::unique_ptr<LocalOp<T>> localBroadcastOp_;

  std::shared_ptr<PHub> pHub;
  std::vector<PLinkKey> reductionKeys;

  typename W::Pointer inbox_;
};

} // namespace gloo
