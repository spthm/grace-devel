#pragma once

#include "grace/cuda/detail/kernels/rng.cuh"

#include "grace/detail/NonCopyable-inl.h"

#include "grace/error.h"
#include "grace/types.h"

namespace grace {

namespace detail {

// Forward declaration so device-side class can friend host-side.
template <typename StateT> class RNGStates;

template <typename StateT>
class RNGDeviceStates
{
public:
    typedef StateT state_type;

private:
    state_type* const _states;
    const size_t _num_states;

    GRACE_HOST RNGDeviceStates(state_type* const states,
                               const size_t num_states);

public:
    GRACE_DEVICE const state_type& load_state() const;
    GRACE_DEVICE void save_state(const state_type& state);

    friend class RNGStates<StateT>;
};

template <typename StateT>
class RNGStates : private NonCopyable<RNGStates<StateT> >
{
public:
    typedef StateT state_type;

private:
    state_type* _states;
    size_t _num_states;
    unsigned long long _seed;
    int _device_id;
    const static int _block_size;

public:
    // Note explicit to prevent implicit type conversion using single-argument
    // and one non-default argument constructors! We do not want
    // int/size_t-to-RNGStates to be a valid implicit conversion!
    //
    // Note that init_states and device_states are/have non-const methods. This
    // container is "logically const", in that it is not possible to modify the
    // states contained within a const RNGStates instance, though they may be
    // accessed.

    GRACE_HOST explicit RNGStates(const unsigned long long seed = 123456789);

    GRACE_HOST explicit RNGStates(const size_t num_states,
                                  const unsigned long long seed = 123456789);

    GRACE_HOST explicit RNGStates(const int device_id,
                                  const unsigned long long seed = 123456789);

    GRACE_HOST RNGStates(const int device_id, const size_t num_states,
                         const unsigned long long seed = 123456789);

    GRACE_HOST ~RNGStates();

    GRACE_HOST void init_states();

    GRACE_HOST void set_seed(const unsigned long long seed);

    GRACE_HOST size_t size() const;

    GRACE_HOST size_t size_bytes() const;

    GRACE_HOST RNGDeviceStates<StateT> device_states();

    GRACE_HOST const RNGDeviceStates<StateT> device_states() const;

private:
    GRACE_HOST void set_device_num_states();
};

template <typename StateT>
const int RNGStates<StateT>::_block_size = 128;


//
// RNGDeviceStates method definitions.
//

template <typename StateT>
GRACE_HOST
RNGDeviceStates<StateT>::RNGDeviceStates(state_type* const states,
                                         const size_t num_states)
    : _states(states), _num_states(num_states) {}

template <typename StateT>
GRACE_DEVICE
const StateT& RNGDeviceStates<StateT>::load_state() const
{
    // Assume no higher dimensionality than 1D grid of 2D blocks.
    // It's ray tracing.
    unsigned int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    GRACE_ASSERT(tid < _num_states);
    return _states[tid];
}

template <typename StateT>
GRACE_DEVICE
void RNGDeviceStates<StateT>::save_state(const state_type& state)
{
    // Assume no higher dimensionality than 1D grid of 2D blocks.
    // It's ray tracing.
    unsigned int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;;
    GRACE_ASSERT(tid < _num_states);
    _states[tid] = state;
}


//
// RNGStates method definitions.
//

template <typename StateT>
GRACE_HOST
RNGStates<StateT>::RNGStates(const unsigned long long seed)
    : _states(NULL), _seed(seed)
{
    GRACE_CUDA_CHECK(cudaGetDevice(&_device_id));
    set_device_num_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RNGStates<StateT>::RNGStates(const size_t num_states,
                             const unsigned long long seed)
    : _states(NULL), _num_states(num_states), _seed(seed)
{
    GRACE_CUDA_CHECK(cudaGetDevice(&_device_id));
    init_states();
}

template <typename StateT>
GRACE_HOST
RNGStates<StateT>::RNGStates(const int device_id,
                             const unsigned long long seed)
    : _states(NULL), _device_id(device_id), _seed(seed)
{
    set_device_num_states();
    init_states();
}

template <typename StateT>
GRACE_HOST
RNGStates<StateT>::RNGStates(const int device_id, const size_t num_states,
                             const unsigned long long seed)
    : _states(NULL), _device_id(device_id), _num_states(num_states), _seed(seed)
{
    init_states();
}

template <typename StateT>
GRACE_HOST
RNGStates<StateT>::~RNGStates()
{
    GRACE_CUDA_CHECK(cudaFree(_states));
}

template <typename StateT>
GRACE_HOST
void RNGStates<StateT>::init_states()
{
    int cur_device_id;
    GRACE_CUDA_CHECK(cudaGetDevice(&cur_device_id));

    GRACE_CUDA_CHECK(cudaSetDevice(_device_id));
    GRACE_CUDA_CHECK(cudaFree(_states));
    cudaError_t err = cudaMalloc((void**)(&_states),
                                 _num_states * sizeof(state_type));
    GRACE_CUDA_CHECK(err);
    const int num_blocks = (_num_states + _block_size - 1) / _block_size;
    init_PRNG_states_kernel<<<num_blocks, _block_size>>>(
        _states, _seed, _num_states);

    GRACE_CUDA_CHECK(cudaSetDevice(cur_device_id));
}

template <typename StateT>
GRACE_HOST
void RNGStates<StateT>::set_seed(const unsigned long long seed)
{
    _seed = seed;
}

template <typename StateT>
GRACE_HOST
size_t RNGStates<StateT>::size() const
{
    return _num_states;
}

template <typename StateT>
GRACE_HOST
size_t RNGStates<StateT>::size_bytes() const
{
    return _num_states * sizeof(state_type);
}

template <typename StateT>
GRACE_HOST
void RNGStates<StateT>::set_device_num_states()
{
    cudaDeviceProp dp;
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&dp, _device_id));
    GRACE_CUDA_CHECK(cudaSetDevice(_device_id));
    _num_states = dp.multiProcessorCount * dp.maxThreadsPerMultiProcessor;
}

template <typename StateT>
GRACE_HOST
RNGDeviceStates<StateT> RNGStates<StateT>::device_states()
{
    return RNGDeviceStates<StateT>(_states, _num_states);
}

template <typename StateT>
GRACE_HOST
const RNGDeviceStates<StateT> RNGStates<StateT>::device_states() const
{
    return RNGDeviceStates<StateT>(_states, _num_states);
}


//
// Convenience typedefs
//

typedef RNGStates<curandStatePhilox4_32_10_t> PRNGStates;
typedef RNGDeviceStates<curandStatePhilox4_32_10_t> PRNGDeviceStates;

} // namespace detail

} // namespace grace
