#pragma once

#include "grace/cuda/detail/kernels/rng.cuh"

#include "grace/error.h"
#include "grace/types.h"

namespace grace {


//
// Forward declarations.
//

namespace detail {

size_t max_device_threads(void);

} // namespace detail


//
// Misc
///

template <typename StateT>
const int RngStates<StateT>::block_size_ = 128;


//
// RngDeviceStates method definitions.
//

template <typename StateT>
GRACE_HOST
RngDeviceStates<StateT>::RngDeviceStates(state_type* const states,
                                         const size_t num_states)
    : states_(states), num_states_(num_states) {}

template <typename StateT>
GRACE_HOST_DEVICE
size_t RngDeviceStates<StateT>::size() const
{
    return num_states_;
}

template <typename StateT>
GRACE_DEVICE
const StateT& RngDeviceStates<StateT>::load_state() const
{
    // Assume no higher dimensionality than 1D grid of 2D blocks.
    // It's ray tracing.
    unsigned int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    GRACE_ASSERT(tid < num_states_);
    return states_[tid];
}

template <typename StateT>
GRACE_DEVICE
void RngDeviceStates<StateT>::save_state(const state_type& state)
{
    // Assume no higher dimensionality than 1D grid of 2D blocks.
    // It's ray tracing.
    unsigned int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;;
    GRACE_ASSERT(tid < num_states_);
    states_[tid] = state;
}


//
// RngStates method definitions.
//

template <typename StateT>
GRACE_HOST
RngStates<StateT>::RngStates(const unsigned long long seed)
    : states_(NULL), seed_(seed)
{
    num_states_ = detail::max_device_threads();
    alloc_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RngStates<StateT>::RngStates(const size_t num_states,
                             const unsigned long long seed)
    : states_(NULL), num_states_(num_states), seed_(seed)
{
    alloc_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RngStates<StateT>::~RngStates()
{
    GRACE_CUDA_CHECK(cudaFree(states_));
}

template <typename StateT>
GRACE_HOST
void RngStates<StateT>::init_states()
{
    const int num_blocks = (num_states_ + block_size_ - 1) / block_size_;
    init_PRNG_states_kernel<<<num_blocks, block_size_>>>(
        states_, seed_, num_states_);
}

template <typename StateT>
GRACE_HOST
void RngStates<StateT>::init_states(const unsigned long long seed)
{
    set_seed(seed);
    init_states();
}

template <typename StateT>
GRACE_HOST
void RngStates<StateT>::set_seed(const unsigned long long seed)
{
    seed_ = seed;
}

template <typename StateT>
GRACE_HOST
size_t RngStates<StateT>::size() const
{
    return num_states_;
}

template <typename StateT>
GRACE_HOST
size_t RngStates<StateT>::size_bytes() const
{
    return num_states_ * sizeof(state_type);
}

template <typename StateT>
GRACE_HOST
RngDeviceStates<StateT> RngStates<StateT>::device_states()
{
    return RngDeviceStates<StateT>(states_, num_states_);
}

template <typename StateT>
GRACE_HOST
const RngDeviceStates<StateT> RngStates<StateT>::device_states() const
{
    return RngDeviceStates<StateT>(states_, num_states_);
}

template <typename StateT>
GRACE_HOST
void RngStates<StateT>::alloc_states()
{
    if (states_ != NULL) {
        GRACE_CUDA_CHECK(cudaFree(states_));
    }
    cudaError_t err = cudaMalloc((void**)(&states_),
                                 num_states_ * sizeof(state_type));
    GRACE_CUDA_CHECK(err);
}


//
// Helper function definitions
//

namespace detail {

size_t max_device_threads()
{
    int device_id;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp dp;
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&dp, device_id));
    return dp.multiProcessorCount * dp.maxThreadsPerMultiProcessor;
}

} // namespace detail

} // namespace grace
