#pragma once

#include "grace/execution_tag.h"
#include "grace/cpp/bvh.h"
#include "grace/detail/bvh_base.h"
#include "grace/cuda/detail/bvh_node.cuh"

#include <thrust/device_vector.h>

namespace grace {

namespace detail {

typedef BvhBase<thrust::device_vector<CudaBvhNode>,
                thrust::device_vector<CudaBvhLeaf> > CudaBvhBase;

} // namespace detail


template <>
class Bvh<grace::cuda_tag> : public detail::CudaBvhBase
{
public:
    GRACE_HOST
    explicit Bvh<grace::cuda_tag>(const size_t num_primitives,
                                  const int max_per_leaf = 1)
        : detail::CudaBvhBase(num_primitives, max_per_leaf) {}

    void from_host(const HostBvh& bvh);
    void to_host(HostBvh& bvh) const;
};

typedef Bvh<grace::cuda_tag> CudaBvh;

} //namespace grace

#include "grace/cuda/detail/bvh-inl.cuh"
