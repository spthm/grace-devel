#pragma once

#include "grace/cuda/CudaBVH.h"

namespace grace {

//
// Constructors
//

template <typename PrimitiveType>
GRACE_HOST
CudaBvh<PrimitiveType>::CudaBVH(const size_t N_primitives,
                                const int max_per_leaf) :
    max_per_leaf(max_per_leaf), root_index(-1)
{
    _primitives.reserve(N_primitives);
    reserve_nodes();
}

template <typename PrimitiveType>
GRACE_HOST
CudaBvh<PrimitiveType>::CudaBVH(const PrimitiveType* host_ptr,
                                const size_t N_primitives,
                                const int max_per_leaf) :
    _primitives(host_ptr, host_ptr + N_primitives), max_per_leaf(max_per_leaf),
    root_index(-1)
{
    reserve_nodes();
}

template <typename PrimitiveType>
GRACE_HOST
CudaBvh<PrimitiveType>::CudaBVH(const std::vector<PrimitiveType>& primitives,
                                const int max_per_leaf) :
    _primitives(primitives), max_per_leaf(max_per_leaf), root_index(-1)
{
    reserve_nodes();
}

template <typename PrimitiveType>
GRACE_HOST
CudaBvh<PrimitiveType>::CudaBVH(const thrust::host_vector<PrimitiveType>& primitives,
                                const int max_per_leaf) :
    _primitives(primitives), max_per_leaf(max_per_leaf), root_index(-1)
{
    reserve_nodes();
}

template <typename PrimitiveType>
GRACE_HOST
CudaBVH<PrimitiveType>::CudaBVH(const thrust::device_ptr<PrimitiveType> primitives,
                                const size_t N_primitives,
                                const int max_per_leaf) :
    _primitives(primitives, primitives + N_primitives),
    max_per_leaf(max_per_leaf), root_index(-1)
{
    reserve_nodes();
}

template <typename PrimitiveType>
GRACE_HOST
CudaBvh<PrimitiveType>::CudaBVH(const thrust::device_vector<PrimitiveType>& primitives,
                                const int max_per_leaf) :
    _primitives(primitives), max_per_leaf(max_per_leaf), root_index(-1)
{
    reserve_nodes();
}

template <typename PrimitiveType>
GRACE_HOST
CudaBvh<PrimitiveType>::CudaBVH(PrimitiveIter first, PrimitiveIter last,
                                const int max_per_leaf) :
    _primitives(first, last), max_per_leaf(max_per_leaf), root_index(-1)
{
    reserve_nodes();
}

template <typename PrimitiveType>
GRACE_HOST
CudaBvh<PrimitiveType>::CudaBVH(const CudaBVH<PrimitiveIter>& other) :
    _primitives(other.primitives), _nodes(other.nodes), _leaves(other.leaves),
    max_per_leaf(other.max_per_leaf), root_index(other.root_index) {}


//
// Public member functions
//

template <typename PrimitiveType>
GRACE_HOST const CudaBVH<PrimitiveType>::primitive_vector&
CudaBvh<PrimitiveType>::primitives() const
{
    return _primitives;
}

template <typename PrimitiveType>
GRACE_HOST CudaBVH<PrimitiveType>::primitive_vector&
CudaBvh<PrimitiveType>::primitives()
{
    return _primitives;
}


//
// Private member functions
//

template <typename PrimitiveType>
GRACE_HOST const CudaBvh<PrimitiveType>::node_vector&
CudaBVH<PrimitiveType>::nodes() const
{
    return _nodes;
}

template <typename PrimitiveType>
GRACE_HOST CudaBvh<PrimitiveType>::node_vector&
CudaBVH<PrimitiveType>::nodes()
{
    return _nodes;
}

template <typename PrimitiveType>
GRACE_HOST const CudaBvh<PrimitiveType>::leaf_vector&
CudaBVH<PrimitiveTypes>::leaves() const
{
    return _leaves;
}

template <typename PrimitiveType>
GRACE_HOST CudaBvh<PrimitiveType>::leaf_vector&
CudaBVH<PrimitiveTypes>::leaves()
{
    return _leaves;
}

template <typename PrimitiveType>
GRACE_HOST void
CudaBvh<PrimitiveType>::reserve_nodes()
{
    size_t estimate = _primitives.size();
    if (max_per_leaf > 1) {
        estimate = (size_t)(1.4 * (_primitives.size() / max_per_leaf));
    }

    _nodes.reserve(esimate);
    _leaves.reserve(esimate);
}


} //namespace grace
