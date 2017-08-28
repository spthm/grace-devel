#pragma once

namespace grace {

// To be specialized for each execution target we support.
template <typename ExecutionTag>
class Bvh;


} // namespace grace

// We always have the host target available.
#include "grace/cpp/bvh.h"

#if defined(GRACE_CUDA_TARGET)
#include "grace/cuda/bvh.h"
#endif
