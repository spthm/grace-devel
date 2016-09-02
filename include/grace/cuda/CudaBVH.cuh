#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace grace {

namespace detail {

// Forward declaration
class CudaBVHPtrs;

} // namespace detail

class CudaBVH
{
public:
    const int max_per_leaf;
    int root_index;
    int* d_root_index_ptr;

    // Allocates space only.
    GRACE_HOST CudaBVH(const size_t N_primitives, const int max_per_leaf = 1);

    // Copies all BVH data.
    GRACE_HOST CudaBVH(const CudaBVH& other);

    GRACE_HOST size_t num_nodes() const;
    GRACE_HOST size_t num_leaves() const;

    GRACE_HOST ~CudaBVH();

private:
    typedef typename thrust::device_vector<detail::CudaNode> node_vector;
    typedef typename thrust::device_vector<detail::CudaLeaf> leaf_vector;

    node_vector _nodes;
    leaf_vector _leaves;

    GRACE_HOST void reserve_nodes();

    friend class detail::CudaBVHPtrs;
};

} //namespace grace

#include "grace/cuda/detail/CudaBVH-inl.h"
#include "grace/cuda/detail/CudaBVHPtrs-inl.h"
