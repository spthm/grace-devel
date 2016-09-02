#pragma once

#include "grace/cuda/CudaBVH.h"

namespace grace {

//
// Constructors
//

GRACE_HOST
CudaBVH::CudaBVH(const size_t N_primitives, const int max_per_leaf) :
    max_per_leaf(max_per_leaf), root_index(-1)
{
    _primitives.reserve(N_primitives);
    reserve_nodes();

    GRACE_CUDA_CHECK(cudaMalloc(&d_root_index_ptr, sizeof(int)));
    GRACE_CUDA_CHECK(cudaMemcpy((void*)d_root_index_ptr,
                                (const void*)&root_index,
                                sizeof(int), cudaMemcpyHostToDevice));
}

GRACE_HOST
CudaBVH::CudaBVH(const CudaBVH& other) :
    _nodes(other.nodes), _leaves(other.leaves),
    max_per_leaf(other.max_per_leaf), root_index(other.root_index)
{
    GRACE_CUDA_CHECK(cudaMalloc(&d_root_index_ptr, sizeof(int)));
    GRACE_CUDA_CHECK(cudaMemcpy((void*)d_root_index_ptr,
                                (const void*)&root_index,
                                sizeof(int), cudaMemcpyHostToDevice));
}


//
// Destructor
//
GRACE_HOST
CudaBVH::~CudaBVH()
{
    GRACE_CUDA_CHECK(cudaFree(d_root_index_ptr));
}


//
// Public member functions
//

GRACE_HOST
size_t CudaBVH::num_nodes() const
{
    return _nodes.size();
}

GRACE_HOST CudaBVH::num_leaves() const
{
    return _leaves.size();
}


//
// Private member functions
//

GRACE_HOST
void CudaBVH::reserve_nodes()
{
    size_t estimate = _primitives.size();
    if (max_per_leaf > 1) {
        estimate = (size_t)(1.4 * (_primitives.size() / max_per_leaf));
    }

    _nodes.reserve(esimate);
    _leaves.reserve(esimate);
}


} //namespace grace
