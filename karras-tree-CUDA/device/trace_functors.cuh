#pragma once

#include "../error.h"
#include "../types.h"

#include "intersect.cuh"
#include "util.cuh"

namespace grace
{

// RayData structs.

template<typename T>
struct RayData_datum
{
    T data;
};

template <typename T, typename Real>
struct RayData_sphere
{
    T data;
    Real b2, dist;
};


namespace gpu {

// 'Null' functors, when functionality is not required.
// Only makes sense for Init(), RayEntry() and RayExit().

class Init_null
{
public:
    GRACE_DEVICE void operator()(const UserSmemPtr<char>& sm_ptr)
    {
        return;
    }
};

class RayEntry_null
{
public:
    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 const RayData& ray_data,
                                 const UserSmemPtr<char>& ptr)
    {
        return;
    }
};

typedef RayEntry_null RayExit_null;


// RayEntry functors.

class RayEntry_zero
{
public:
    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        ray_data.data = 0;
    }
};

template <typename T>
class RayEntry_from_array
{
private:
    const T* const inits;

public:
    RayEntry_from_array(const T* const ray_data_inits) : inits(ray_data_inits) {}

    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        ray_data.data = inits[ray_idx];
    }
};


// RayExit functors.

template <typename T>
class RayExit_to_array
{
private:
    T* const store;

public:
    RayExit_to_array(T* const ray_data_store) : store(ray_data_store) {}

    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 const RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        store[ray_idx] = ray_data.data;
    }
};


// Init functors.

// Copying a contiguous set of data to SMEM.
template <typename T>
class InitGlobalToSmem
{
private:
    const T* const data_global;
    const int count;

public:
    InitGlobalToSmem(const T* const global_addr, const int count) :
        data_global(global_addr), count(count) {}

    GRACE_DEVICE void operator()(const UserSmemPtr<char>& sm_ptr)
    {
        // We *must* cast from the default pointer-to-char to the data type we
        // wish to store in shared memory.
        UserSmemPtr<T> T_ptr = sm_ptr;

        for (int i = threadIdx.x; i < count; i += blockDim.x)
        {
            T_ptr[i] = data_global[i];
        }

        // __syncthreads() is called by the trace kernel.
    }
};


// Intersection functors.

// Discards impact parameter squared and distance to intersection.
class Intersect_sphere_bool
{
public:
    template <typename Real4, typename RayData>
    GRACE_DEVICE bool operator()(const Ray& ray, const Real4& sphere,
                                 const RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        typedef typename Real4ToRealMapper<Real4>::type Real;

        Real dummy_b2, dummy_dist;
        return sphere_hit(ray, sphere, dummy_b2, dummy_dist);
    }
};

// Stores impact parameter squared and distance to intersection.
class Intersect_sphere_b2dist
{
public:
    template <typename Real4, typename RayData>
    GRACE_DEVICE bool operator()(const Ray& ray, const Real4& sphere,
                                 RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        return sphere_hit(ray, sphere, ray_data.b2, ray_data.dist);
    }
};


// OnHit functors.

class OnHit_increment
{
public:
    template <typename RayData, typename TPrim>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData& ray_data,
                                 const int prim_idx, const TPrim& primitive,
                                 const UserSmemPtr<char>& sm_ptr)
    {
        ++ray_data.data;
    }
};

// Accumulating per-ray kernel integrals.
class OnHit_sphere_cumulative
{
private:
    const int N_table;

public:
    OnHit_sphere_cumulative(const int N_table) : N_table(N_table) {}

    template <typename RayData, typename Real4>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData& ray_data,
                                 const int sphere_idx, const Real4& sphere,
                                 const UserSmemPtr<char>& sm_ptr)
    {
        typedef typename Real4ToRealMapper<Real4>::type Real;

        // For simplicity, we do not template the type of the kernel integral
        // lookup table; it is always required to be double.
        UserSmemPtr<double> Wk_lookup = sm_ptr;

        Real ir = 1.f / sphere.w;
        Real b = (N_table - 1) * (sqrt(ray_data.b2) * ir);
        Real integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data.dist >= 0);

        ray_data.data += integral;
    }
};

// Storing per-ray kernel integrals, sphere indices and ray-particle distances.
template <typename IntegerIdx, typename Real>
class OnHit_sphere_individual
{
private:
    IntegerIdx* const indices;
    Real* const integrals;
    Real* const distances;
    const int N_table;

public:
    OnHit_sphere_individual(IntegerIdx* const indices, Real* const integrals,
                            Real* const distances, const int N_table) :
        indices(indices), integrals(integrals), distances(distances),
        N_table(N_table) {}

    template <typename RayData, typename Real4>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData& ray_data,
                                 const int sphere_idx, const Real4& sphere,
                                 const UserSmemPtr<char>& sm_ptr)
    {
        GRACE_ASSERT(are_types_equal<Real>(sphere.x));

        // For simplicity, we do not template the type of the kernel integral
        // lookup table; it is always required to be double.
        UserSmemPtr<double> Wk_lookup = sm_ptr;

        Real ir = 1.f / sphere.w;
        Real b = (N_table - 1) * (sqrt(ray_data.b2) * ir);
        Real integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        indices[ray_data.data] = sphere_idx;
        integrals[ray_data.data] = integral;
        distances[ray_data.data] = ray_data.dist;

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data.dist >= 0);

        ++ray_data.data;
    }
};

} // namespace gpu

} // namespace grace
