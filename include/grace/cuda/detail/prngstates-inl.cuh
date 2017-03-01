#pragma once

#include "grace/cuda/detail/kernels/rng.cuh"

#include "grace/detail/noncopyable-inl.h"

#include "grace/error.h"
#include "grace/types.h"

namespace grace {

namespace detail {

//
// Forward declarations.
//

size_t max_device_threads(const int device_id);
// So device-side class can friend host-side.
template <typename StateT> class RngStates;


//
// Class declarations
//

template <typename StateT>
class RngDeviceStates
{
public:
    typedef StateT state_type;

private:
    state_type* const _states;
    const size_t _num_states;

    GRACE_HOST RngDeviceStates(state_type* const states,
                               const size_t num_states);

public:
    GRACE_DEVICE const state_type& load_state() const;
    GRACE_DEVICE void save_state(const state_type& state);

    friend class RngStates<StateT>;
};

// A class to initialize states on the current device, or, if provided,
// any device. The target device cannot be modified after initialization.
// Resource allocation happens at initialization, and only at initialization.
// All state initialization always occurs on the original device; the current
// device is _temporarily_ set to the RngStates' device if necessary.
template <typename StateT>
class RngStates : private NonCopyable<RngStates<StateT> >
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
    // int/size_t-to-RngStates to be a valid implicit conversion!
    //
    // Note that init_states and device_states are/have non-const methods. This
    // container is "logically const", in that it is not possible to modify the
    // states contained within a const RngStates instance, though they may be
    // accessed.

    GRACE_HOST explicit RngStates(const unsigned long long seed = 123456789);

    GRACE_HOST explicit RngStates(const size_t num_states,
                                  const unsigned long long seed = 123456789);

    GRACE_HOST explicit RngStates(const int device_id,
                                  const unsigned long long seed = 123456789);

    GRACE_HOST RngStates(const int device_id, const size_t num_states,
                         const unsigned long long seed = 123456789);

    GRACE_HOST ~RngStates();

    GRACE_HOST int device() const;

    GRACE_HOST void init_states();

    GRACE_HOST void init_states(const unsigned long long seed);

    GRACE_HOST void set_seed(const unsigned long long seed);

    GRACE_HOST size_t size() const;

    GRACE_HOST size_t size_bytes() const;

    GRACE_HOST RngDeviceStates<StateT> device_states();

    GRACE_HOST const RngDeviceStates<StateT> device_states() const;

private:
    GRACE_HOST void alloc_states();
    GRACE_HOST int swap_device() const;
    GRACE_HOST void unswap_device(const int device) const;
};

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
    _num_states = max_device_threads(_device_id);
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
    _num_states = max_device_threads(_device_id);
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
// Convenience typedefs
//

typedef RngStates<curandStateXORWOW_t> PrngStates;
typedef RngDeviceStates<curandStateXORWOW_t> PrngDeviceStates;


//
// Helper function definitions
//

size_t max_device_threads(const int device_id)
{
    cudaDeviceProp dp;
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&dp, device_id));
    return dp.multiProcessorCount * dp.maxThreadsPerMultiProcessor;
}

} // namespace detail

} // namespace grace
