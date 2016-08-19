#pragma once

#include "grace/generic/meta.h"
#include "grace/sphere.h"

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

template <typename Real3>
class random_real3_functor
{
    typedef typename grace::Real3ToRealMapper<Real3>::type Real;

    thrust::uniform_real_distribution<Real> x_uniform;
    thrust::uniform_real_distribution<Real> y_uniform;
    thrust::uniform_real_distribution<Real> z_uniform;

public:
    random_real3_functor() :
        x_uniform(0.0, 1.0),
        y_uniform(0.0, 1.0),
        z_uniform(0.0, 1.0) {}

    explicit random_real3_functor(const Real high) :
        x_uniform(0.0, high),
        y_uniform(0.0, high),
        z_uniform(0.0, high) {}

    explicit random_real3_functor(const Real low, const Real high) :
        x_uniform(low, high),
        y_uniform(low, high),
        z_uniform(low, high) {}

    explicit random_real3_functor(const Real3 high) :
        x_uniform(0.0, high.x),
        y_uniform(0.0, high.y),
        z_uniform(0.0, high.z) {}

    explicit random_real3_functor(const Real3 low,
                                   const Real3 high) :
        x_uniform(low.x, high.x),
        y_uniform(low.y, high.y),
        z_uniform(low.z, high.z) {}

    GRACE_HOST_DEVICE Real3 operator()(unsigned int n)
    {
        unsigned int seed = hash(n);

        thrust::default_random_engine rng(seed);

        Real3 xyz;
        xyz.x = x_uniform(rng);
        xyz.y = y_uniform(rng);
        xyz.z = z_uniform(rng);

        return xyz;
    }
};

template <typename SphereType>
class random_sphere_functor
{
    typedef typename SphereType::value_type Real;

    thrust::uniform_real_distribution<Real> x_uniform;
    thrust::uniform_real_distribution<Real> y_uniform;
    thrust::uniform_real_distribution<Real> z_uniform;
    thrust::uniform_real_distribution<Real> r_uniform;

public:
    random_sphere_functor() :
        x_uniform(0.0, 1.0),
        y_uniform(0.0, 1.0),
        z_uniform(0.0, 1.0),
        r_uniform(0.0, 1.0) {}

    explicit random_sphere_functor(const Real high) :
        x_uniform(0.0, high),
        y_uniform(0.0, high),
        z_uniform(0.0, high),
        r_uniform(0.0, high) {}

    explicit random_sphere_functor(const Real low, const Real high) :
        x_uniform(low, high),
        y_uniform(low, high),
        z_uniform(low, high),
        r_uniform(low, high) {}

    explicit random_sphere_functor(const SphereType sphere_high) :
        x_uniform(0.0, sphere_high.x),
        y_uniform(0.0, sphere_high.y),
        z_uniform(0.0, sphere_high.z),
        r_uniform(0.0, sphere_high.r) {}

    explicit random_sphere_functor(const SphereType sphere_low,
                                   const SphereType sphere_high) :
        x_uniform(sphere_low.x, sphere_high.x),
        y_uniform(sphere_low.y, sphere_high.y),
        z_uniform(sphere_low.z, sphere_high.z),
        r_uniform(sphere_low.r, sphere_high.r) {}

    GRACE_HOST_DEVICE SphereType operator()(unsigned int n)
    {
        unsigned int seed = hash(n);

        thrust::default_random_engine rng(seed);

        SphereType sphere;
        sphere.x = x_uniform(rng);
        sphere.y = y_uniform(rng);
        sphere.z = z_uniform(rng);
        sphere.r = r_uniform(rng);

        return sphere;
    }
};
