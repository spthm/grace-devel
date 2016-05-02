#pragma once

#include "grace/generic/util/meta.h"

#include <thrust/random.h>

//-----------------------------------------------------------------------------
// Utilities for generating random floats
//-----------------------------------------------------------------------------

// This hash function is due to Thomas Wang and Bob Jenkins.
// See e.g.,
// http://www.burtleburtle.net/bob/hash/integer.html
// http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
//
// It is also used in the Thrust monte_carlo.cu example,
// https://github.com/thrust/thrust/blob/master/examples/monte_carlo.cu
// Thrust also provides a more robust alternative,
// https://github.com/thrust/thrust/blob/master/examples/monte_carlo_disjoint_sequences.cu
GRACE_HOST_DEVICE unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

template <typename Real>
class random_real_functor
{
    thrust::uniform_real_distribution<Real> uniform;

public:
    random_real_functor() :
        uniform(0.0, 1.0) {}

    explicit random_real_functor(const Real high) :
        uniform(0.0, high) {}

    explicit random_real_functor(const Real low, const Real high) :
        uniform(low, high) {}

    GRACE_HOST_DEVICE Real operator()(unsigned int n)
    {
        unsigned int seed = hash(n);

        thrust::default_random_engine rng(seed);

        return uniform(rng);
    }
};

template <typename Real4>
class random_real4_functor
{
    typedef typename grace::Real4ToRealMapper<Real4>::type Real;

    thrust::uniform_real_distribution<Real> x_uniform;
    thrust::uniform_real_distribution<Real> y_uniform;
    thrust::uniform_real_distribution<Real> z_uniform;
    thrust::uniform_real_distribution<Real> w_uniform;

public:
    random_real4_functor() :
        x_uniform(0.0, 1.0),
        y_uniform(0.0, 1.0),
        z_uniform(0.0, 1.0),
        w_uniform(0.0, 1.0) {}

    explicit random_real4_functor(const Real high) :
        x_uniform(0.0, high),
        y_uniform(0.0, high),
        z_uniform(0.0, high),
        w_uniform(0.0, high) {}

    explicit random_real4_functor(const Real low, const Real high) :
        x_uniform(low, high),
        y_uniform(low, high),
        z_uniform(low, high),
        w_uniform(low, high) {}

    explicit random_real4_functor(const Real4 xyzw_high) :
        x_uniform(0.0, xyzw_high.x),
        y_uniform(0.0, xyzw_high.y),
        z_uniform(0.0, xyzw_high.z),
        w_uniform(0.0, xyzw_high.w) {}

    explicit random_real4_functor(const Real4 xyzw_low, const Real4 xyzw_high) :
        x_uniform(xyzw_low.x, xyzw_high.x),
        y_uniform(xyzw_low.y, xyzw_high.y),
        z_uniform(xyzw_low.z, xyzw_high.z),
        w_uniform(xyzw_low.w, xyzw_high.w) {}

    GRACE_HOST_DEVICE Real4 operator()(unsigned int n)
    {
        unsigned int seed = hash(n);

        thrust::default_random_engine rng(seed);

        Real4 xyzw;
        xyzw.x = x_uniform(rng);
        xyzw.y = y_uniform(rng);
        xyzw.z = z_uniform(rng);
        xyzw.w = w_uniform(rng);

        return xyzw;
    }
};
