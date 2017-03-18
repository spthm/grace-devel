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
    state_type* const states_;
    const size_t num_states_;

    GRACE_HOST RngDeviceStates(state_type* const states,
                               const size_t num_states);

public:
    GRACE_HOST_DEVICE size_t size() const;
    GRACE_DEVICE const state_type& load_state() const;
    GRACE_DEVICE void save_state(const state_type& state);

    // No one else should be able to construct an RngDeviceStates.
    friend class RngStates<StateT>;
};

// A class to initialize states on the current device.
// Resource allocation happens at initialization, and only at initialization.
template <typename StateT>
class RngStates : private detail::NonCopyable<RngStates<StateT> >
{
public:
    typedef StateT state_type;

private:
    state_type* states_;
    size_t num_states_;
    unsigned long long seed_;
    const static int block_size_;

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
                                  const unsigned long long seed);

    GRACE_HOST ~RngStates();

    GRACE_HOST void init_states();

    GRACE_HOST void init_states(const unsigned long long seed);

    GRACE_HOST void set_seed(const unsigned long long seed);

    GRACE_HOST size_t size() const;

    GRACE_HOST size_t size_bytes() const;

    GRACE_HOST RngDeviceStates<StateT> device_states();

    GRACE_HOST const RngDeviceStates<StateT> device_states() const;

private:
    GRACE_HOST void alloc_states();
};


//
// Convenience typedefs
//

typedef RngStates<curandStateXORWOW_t> PrngStates;
typedef RngDeviceStates<curandStateXORWOW_t> PrngDeviceStates;

} // namespace grace

#include "grace/cuda/detail/prngstates-inl.cuh"
