#pragma once

#include "grace/cuda/detail/kernels/rng.cuh"

#include "grace/error.h"
#include "grace/types.h"

namespace grace {


//
// Forward declarations.
//

namespace detail {

size_t max_device_threads(const int device_id);

} // namespace detail


//
// Misc
///

template <typename StateT>
const int RngStates<StateT>::_block_size = 128;


//
// RngDeviceStates method definitions.
//

template <typename StateT>
GRACE_HOST
RngDeviceStates<StateT>::RngDeviceStates(state_type* const states,
                                         const size_t num_states)
    : _states(states), _num_states(num_states) {}

template <typename StateT>
GRACE_HOST_DEVICE
size_t RngDeviceStates<StateT>::size() const
{
    return _num_states;
}

template <typename StateT>
GRACE_DEVICE
const StateT& RngDeviceStates<StateT>::load_state() const
{
    // Assume no higher dimensionality than 1D grid of 2D blocks.
    // It's ray tracing.
    unsigned int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    GRACE_ASSERT(tid < _num_states);
    return _states[tid];
}

template <typename StateT>
GRACE_DEVICE
void RngDeviceStates<StateT>::save_state(const state_type& state)
{
    // Assume no higher dimensionality than 1D grid of 2D blocks.
    // It's ray tracing.
    unsigned int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;;
    GRACE_ASSERT(tid < _num_states);
    _states[tid] = state;
}


//
// RngStates method definitions.
//

template <typename StateT>
GRACE_HOST
RngStates<StateT>::RngStates(const unsigned long long seed)
    : _states(NULL), _seed(seed)
{
    GRACE_CUDA_CHECK(cudaGetDevice(&_device_id));
    _num_states = detail::max_device_threads(_device_id);
    alloc_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RngStates<StateT>::RngStates(const size_t num_states,
                             const unsigned long long seed)
    : _states(NULL), _num_states(num_states), _seed(seed)
{
    GRACE_CUDA_CHECK(cudaGetDevice(&_device_id));
    alloc_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RngStates<StateT>::RngStates(const int device_id,
                             const unsigned long long seed)
    : _states(NULL), _device_id(device_id), _seed(seed)
{
    _num_states = detail::max_device_threads(_device_id);
    alloc_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RngStates<StateT>::RngStates(const int device_id, const size_t num_states,
                             const unsigned long long seed)
    : _states(NULL), _device_id(device_id), _num_states(num_states), _seed(seed)
{
    alloc_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RngStates<StateT>::~RngStates()
{
    GRACE_CUDA_CHECK(cudaFree(_states));
}

template <typename StateT>
GRACE_HOST
int RngStates<StateT>::device() const
{
    return _device_id;
}

template <typename StateT>
GRACE_HOST
void RngStates<StateT>::init_states()
{
    int entry_device = swap_device();

    const int num_blocks = (_num_states + _block_size - 1) / _block_size;
    init_PRNG_states_kernel<<<num_blocks, _block_size>>>(
        _states, _seed, _num_states);

    unswap_device(entry_device);
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
    _seed = seed;
}

template <typename StateT>
GRACE_HOST
size_t RngStates<StateT>::size() const
{
    return _num_states;
}

template <typename StateT>
GRACE_HOST
size_t RngStates<StateT>::size_bytes() const
{
    return _num_states * sizeof(state_type);
}

template <typename StateT>
GRACE_HOST
RngDeviceStates<StateT> RngStates<StateT>::device_states()
{
    return RngDeviceStates<StateT>(_states, _num_states);
}

template <typename StateT>
GRACE_HOST
const RngDeviceStates<StateT> RngStates<StateT>::device_states() const
{
    return RngDeviceStates<StateT>(_states, _num_states);
}

template <typename StateT>
GRACE_HOST
void RngStates<StateT>::alloc_states()
{
    int entry_device = swap_device();

    if (_states != NULL) {
        GRACE_CUDA_CHECK(cudaFree(_states));
    }
    cudaError_t err = cudaMalloc((void**)(&_states),
                                 _num_states * sizeof(state_type));
    GRACE_CUDA_CHECK(err);

    unswap_device(entry_device);
}

template <typename StateT>
GRACE_HOST
int RngStates<StateT>::swap_device() const
{
    int cur;
    GRACE_CUDA_CHECK(cudaGetDevice(&cur));
    GRACE_CUDA_CHECK(cudaSetDevice(_device_id));
    return cur;
}

template <typename StateT>
GRACE_HOST
void RngStates<StateT>::unswap_device(const int device) const
{
    GRACE_CUDA_CHECK(cudaSetDevice(device));
}


//
// Helper function definitions
//

namespace detail {

size_t max_device_threads(const int device_id)
{
    cudaDeviceProp dp;
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&dp, device_id));
    return dp.multiProcessorCount * dp.maxThreadsPerMultiProcessor;
}

} // namespace detail

} // namespace grace
