#pragma once

#include "grace/cuda/detail/device/intersect.cuh"

#include "grace/cuda/util/bound_iter.cuh"

#include "grace/generic/interpolate.h"
#include "grace/generic/meta.h"

#include "grace/error.h"
#include "grace/sphere.h"
#include "grace/types.h"

namespace grace {

// 'Null' functors, when functionality is not required.
// Only makes sense for Init(), RayEntry() and RayExit().

class Init_null
{
public:
    GRACE_DEVICE void operator()(const BoundedPtr<char> /*smem_iter*/)
    {
        return;
    }
};

class RayEntry_null
{
public:
    template <typename RayData>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 const RayData&,
                                 const BoundedPtr<char> /*smem_iter*/)
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
    const T* const inits;

public:
    RayEntry_from_array(const T* const ray_data_inits) : inits(ray_data_inits) {}

    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray&,
                                 RayData& ray_data,
                                 const BoundedPtr<char> /*smem_iter*/)
    {
        ray_data.data = inits[ray_idx];
    }
};


// RayExit functors.

template <typename OutType>
class RayExit_to_array
{
private:
    OutType* const store;

public:
    RayExit_to_array(OutType* const ray_data_store) : store(ray_data_store) {}

    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray&,
                                 const RayData& ray_data,
                                 const BoundedPtr<char> /*smem_iter*/)
    {
        store[ray_idx] = ray_data.data;
    }
};


// Init functors.

// Copying a contiguous set of data to SMEM.
template <typename InType>
class InitGlobalToSmem
{
private:
    const InType* const data_global;
    const int count;

public:
    InitGlobalToSmem(const InType* const global_addr, const int count) :
        data_global(global_addr), count(count) {}

    GRACE_DEVICE void operator()(const BoundedPtr<char> smem_iter)
    {
        // We *must* cast from the default pointer-to-char to the data type we
        // wish to store in shared memory for dereferencing and indexing
        // operators to work correctly.
        BoundedPtr<InType> in_iter = smem_iter;

        for (int i = threadIdx.x; i < count; i += blockDim.x)
        {
            in_iter[i] = data_global[i];
        }

        // __syncthreads() is called by the trace kernel.
    }
};


// Intersection functors.

// Discards impact parameter squared and distance to intersection.
template <typename PrecisionType>
class Intersect_sphere_bool
{
public:
    template <typename T, typename RayData>
    GRACE_DEVICE bool operator()(const Ray& ray, const Sphere<T>& sphere,
                                 const RayData&, const int /*lane*/,
                                 const BoundedPtr<char> /*smem_iter*/)
    {
        PrecisionType dummy_b2, dummy_dist;
        return sphere_hit<PrecisionType>(ray, sphere, dummy_b2, dummy_dist);
    }
};

// Stores impact parameter squared and distance to intersection.
template <typename PrecisionType>
class Intersect_sphere_b2dist
{
public:
    template <typename T, typename RayData>
    GRACE_DEVICE bool operator()(const Ray& ray, const Sphere<T>& sphere,
                                 RayData& ray_data, const int /*lane*/,
                                 const BoundedPtr<char> /*smem_iter*/)
    {
        // Type of the b2, dist provided to sphere_hit must match the
        // PrecisionType template parameter.
        PrecisionType tmp_b2, tmp_dist;
        bool result = sphere_hit<PrecisionType>(ray, sphere, tmp_b2, tmp_dist);
        ray_data.b2 = tmp_b2;
        ray_data.dist = tmp_dist;
        return result;
    }
};


// OnHit functors.

class OnHit_increment
{
public:
    template <typename RayData, typename TPrim>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 RayData& ray_data, const int /*prim_idx*/,
                                 const TPrim&, const int /*lane*/,
                                 const BoundedPtr<char> /*smem_iter*/)
    {
        ++ray_data.data;
    }
};

// Accumulating per-ray kernel integrals.
template <typename PrecisionType>
class OnHit_sphere_cumulate
{
private:
    const int N_table;

public:
    OnHit_sphere_cumulate(const int N_table) : N_table(N_table) {}

    template <typename RayData, typename T>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 RayData& ray_data, const int /*sphere_idx*/,
                                 const Sphere<T>& sphere, const int /*lane*/,
                                 const BoundedPtr<char> smem_iter)
    {
        // For implementation simplicity, we do not template the type of the
        // kernel integral lookup table; it is always required to be double.
        BoundedPtr<double> Wk_lookup = smem_iter;

        PrecisionType ir = static_cast<PrecisionType>(1.0) / sphere.r;
        PrecisionType b2 = static_cast<PrecisionType>(ray_data.b2);
        PrecisionType b = (N_table - 1) * (sqrt(b2) * ir);
        PrecisionType integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data.dist >= 0);

        ray_data.data += integral;
    }
};

// Storing per-ray kernel integrals, sphere indices and ray-particle distances.
template <typename PrecisionType, typename IndexType, typename OutType>
class OnHit_sphere_individual
{
private:
    IndexType* const indices;
    OutType* const integrals;
    OutType* const distances;
    const int N_table;

public:
    OnHit_sphere_individual(IndexType* const indices, OutType* const integrals,
                            OutType* const distances, const int N_table) :
        indices(indices), integrals(integrals), distances(distances),
        N_table(N_table) {}

    template <typename RayData, typename T>
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const Ray&,
                                 RayData& ray_data, const int sphere_idx,
                                 const Sphere<T>& sphere, const int /*lane*/,
                                 const BoundedPtr<char> smem_iter)
    {
        // For implementation simplicity, we do not template the type of the
        // kernel integral lookup table; it is always required to be double.
        BoundedPtr<double> Wk_lookup = smem_iter;

        PrecisionType ir = static_cast<PrecisionType>(1.0) / sphere.r;
        PrecisionType b2 = static_cast<PrecisionType>(ray_data.b2);
        PrecisionType b = (N_table - 1) * (sqrt(b2) * ir);
        PrecisionType integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        indices[ray_data.data] = sphere_idx;
        integrals[ray_data.data] = integral;
        distances[ray_data.data] = ray_data.dist;

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data.dist >= 0);

        ++ray_data.data;
    }
};

} // namespace grace
