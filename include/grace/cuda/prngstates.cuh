#pragma once

#include "grace/detail/noncopyable-inl.h"
#include "grace/types.h"

#include <curand_kernel.h>

namespace grace {

//
// Forward declarations
//

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
    GRACE_HOST_DEVICE size_t size() const;
    GRACE_DEVICE const state_type& load_state() const;
    GRACE_DEVICE void save_state(const state_type& state);

    // No one else should be able to construct an RngDeviceStates.
    friend class RngStates<StateT>;
};

// A class to initialize states on the current device, or, if provided,
// any device. The target device cannot be modified after initialization.
// Resource allocation happens at initialization, and only at initialization.
// All state initialization always occurs on the original device; the current
// device is _temporarily_ set to the RngStates' device if necessary.
template <typename StateT>
class RngStates : private detail::NonCopyable<RngStates<StateT> >
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


//
// Convenience typedefs
//

typedef RngStates<curandStateXORWOW_t> PrngStates;
typedef RngDeviceStates<curandStateXORWOW_t> PrngDeviceStates;

} // namespace grace

#include "grace/cuda/detail/prngstates-inl.cuh"
