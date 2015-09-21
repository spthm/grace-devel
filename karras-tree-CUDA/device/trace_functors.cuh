#pragma once

#include "../error.h"
#include "../types.h"

#include "intersect.cuh"
#include "interpolation.cuh"

#include "../util/bound_iter.cuh"
#include "../util/meta.h"

namespace grace {

namespace gpu {

// 'Null' functors, when functionality is not required.
// Only makes sense for Init(), RayEntry() and RayExit().

class Init_null
{
public:
    GRACE_DEVICE void operator()(const BoundIter<char> /*smem_iter*/)
    {
        return;
    }
};

class RayEntry_null
{
public:
    template <typename RayData>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 const RayData* const,
                                 const BoundIter<char> /*smem_iter*/)
    {
        return;
    }
};

typedef RayEntry_null RayExit_null;


// RayEntry functors.

template <typename T>
class RayEntry_from_array
{
private:
    const T* inits;

public:
    void setRayEntryArray(const T* const ray_data_inits)
    {
        inits = ray_data_inits;
    }

    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray&,
                                 RayData* const ray_data,
                                 const BoundIter<char> /*smem_iter*/)
    {
        ray_data->data = inits[ray_idx];
    }
};


// RayExit functors.

template <typename T>
class RayExit_to_array
{
private:
    T* store;

public:
    void setRayExitArray(T* const ray_data_store)
    {
        store = ray_data_store;
    }

    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray&,
                                 const RayData* const ray_data,
                                 const BoundIter<char> /*smem_iter*/)
    {
        store[ray_idx] = ray_data->data;
    }
};


// Init functors.

// Copying a contiguous set of data to SMEM.
template <typename T>
class InitGlobalToSmem
{
private:
    const T* data_global;
    int count;

public:
    // Note that N is a count, not a size in bytes.
    void setSmemArraySource(const T* const global_addr, const int N)
    {
        data_global = global_addr;
        count = N;
    }

    GRACE_DEVICE void operator()(const BoundIter<char> smem_iter)
    {
        // We *must* cast from the default pointer-to-char to the data type we
        // wish to store in shared memory for dereferencing and indexing
        // operators to work correctly.
        BoundIter<T> T_iter = smem_iter;

        for (int i = threadIdx.x; i < count; i += blockDim.x)
        {
            T_iter[i] = data_global[i];
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
    GRACE_DEVICE bool operator()(const int /*ray_idx*/, const Ray& ray,
                                 const RayData* const,
                                 const int /*sphere_idx*/, const Real4& sphere,
                                 const int /*lane*/,
                                 const BoundIter<char> /*smem_iter*/)
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
    GRACE_DEVICE bool operator()(const int /*ray_idx*/, const Ray& ray,
                                 RayData* const ray_data,
                                 const int /*sphere_idx*/, const Real4& sphere,
                                 const int /*lane*/,
                                 const BoundIter<char> /*smem_iter*/)
    {
        return sphere_hit(ray, sphere, ray_data->b2, ray_data->dist);
    }
};


// OnHit functors.

class OnHit_increment
{
public:
    template <typename RayData, typename TPrim>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 RayData* const ray_data,
                                 const int /*prim_idx*/, const TPrim&,
                                 const int /*lane*/,
                                 const BoundIter<char> /*smem_iter*/)
    {
        ++(ray_data->data);
    }
};

// Accumulating per-ray kernel integrals.
class OnHit_sphere_cumulate
{
private:
    int N_table;

public:
    void setLookupTableSize(const int N)
    {
        N_table = N;
    }

    template <typename RayData, typename Real4>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 RayData* const ray_data,
                                 const int /*sphere_idx*/, const Real4& sphere,
                                 const int /*lane*/,
                                 const BoundIter<char> smem_iter)
    {
        typedef typename Real4ToRealMapper<Real4>::type Real;

        // For simplicity, we do not template the type of the kernel integral
        // lookup table; it is always required to be double.
        BoundIter<double> Wk_lookup = smem_iter;

        Real ir = 1.f / sphere.w;
        Real b = (N_table - 1) * (sqrt(ray_data->b2) * ir);
        Real integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data->dist >= 0);

        ray_data->data += integral;
    }
};

// Storing per-ray kernel integrals, sphere indices and ray-particle distances.
template <typename IntegerIdx, typename Real>
class OnHit_sphere_individual
{
private:
    IntegerIdx* indices;
    Real* integrals;
    Real* distances;
    int N_table;

public:
    void setOutputArrays(IntegerIdx* const indices_addr,
                         Real* const integrals_addr,
                         Real* const distances_addr)
    {
        indices = indices_addr;
        integrals = integrals_addr;
        distances = distances_addr;
    }

    void setLookupTableSize(const int N)
    {
        N_table = N;
    }

    template <typename RayData, typename Real4>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 RayData* const ray_data, const int sphere_idx,
                                 const Real4& sphere, const int /*lane*/,
                                 const BoundIter<char> smem_iter)
    {
        GRACE_ASSERT(are_types_equal<Real>(sphere.x));

        // For simplicity, we do not template the type of the kernel integral
        // lookup table; it is always required to be double.
        BoundIter<double> Wk_lookup = smem_iter;

        Real ir = 1.f / sphere.w;
        Real b = (N_table - 1) * (sqrt(ray_data->b2) * ir);
        Real integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        indices[ray_data->data] = sphere_idx;
        integrals[ray_data->data] = integral;
        distances[ray_data->data] = ray_data->dist;

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data->dist >= 0);

        ++(ray_data->data);
    }
};

} // namespace gpu

} // namespace grace
