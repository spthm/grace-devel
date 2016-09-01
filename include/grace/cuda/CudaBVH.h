#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace grace {

namespace detail {

// Forward declaration
template <typename PrimitiveType>
class CudaBVHPtrs;

} // namespace detail

template <typename PrimitiveType>
class CudaBVH
{
public:
    typedef PrimitiveType primitive_type;
    typedef typename thrust::device_vector<PrimitiveType> primitive_vector;

    const int max_per_leaf;
    int root_index;

    // Allocates space only.
    GRACE_HOST CudaBVH(const size_t N_primitives, const int max_per_leaf = 1);

    // Copies primitives to device.
    GRACE_HOST CudaBVH(const PrimitiveType* host_ptr, const size_t N_primitives,
                       const int max_per_leaf = 1);

    // Copies primitives to device.
    GRACE_HOST CudaBVH(const std::vector<PrimitiveType>& primitives,
                       const int max_per_leaf = 1);

    // Copies primitives to device.
    GRACE_HOST CudaBVH(const thrust::host_vector<PrimitiveType>& primitives,
                       const int max_per_leaf = 1);

    // Copies primitives.
    GRACE_HOST CudaBVH(const thrust::device_ptr<PrimitiveType> primitives,
                       const size_t N_primitives,
                       const int max_per_leaf = 1);

    // Copies primitives.
    GRACE_HOST CudaBVH(const thrust::device_vector<PrimitiveType>& primitives,
                       const int max_per_leaf = 1);

    // Copies primitives (whether iterator is host- or device-side).
    // Note that if first and last refer to device-side data, they must be
    // thrust iterators or thrust::device_ptrs.
    GRACE_HOST CudaBVH(PrimitiveIter first, PrimitiveIter last,
                       const int max_per_leaf = 1);

    // Copies all BVH data, including primitives.
    GRACE_HOST CudaBVH(const CudaBVH<PrimitiveIter>& other);

    GRACE_HOST const primitive_vector& primitives() const;

    // Care should be taken with this accessor. If the underlying data is
    // modified, the tree must be rebuilt.
    GRACE_HOST primitive_vector& primitives();

private:
    typedef typename thrust::device_vector<detail::CudaNode> node_vector;
    typedef typename thrust::device_vector<detail::CudaLeaf> leaf_vector;

    primitive_vector _primitives;
    node_vector _nodes;
    leaf_vector _leaves;

    GRACE_HOST const leaf_vector& leaves() const;
    GRACE_HOST leaf_vector& leaves();
    GRACE_HOST const node_vector& nodes() const;
    GRACE_HOST node_vector& nodes();
    GRACE_HOST void reserve_nodes();

    friend class detail::CudaBVHPtrs<PrimitiveType>;
};

} //namespace grace

#include "grace/cuda/detail/CudaBVH-inl.h"
#include "grace/cuda/detail/CudaBVHPtrs-inl.h"
