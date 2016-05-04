#pragma once

#include "grace/error.h"
#include "grace/types.h"

// CUDA math constants.
#include <math_constants.h>

#include <iterator>
#include <limits>

namespace grace {

//-----------------------------------------------------------------------------
// GPU helper functions for tree build kernels.
//-----------------------------------------------------------------------------

struct DeltaXOR
{
    GRACE_HOST_DEVICE uinteger32 operator()(
        const int i,
        const uinteger32* morton_keys,
        const size_t n_keys) const
    {
        // delta(-1) and delta(N-1) must return e.g. UINT_MAX because they
        // cover invalid ranges but are valid queries during tree construction.
        if (i < 0 || i + 1 >= n_keys)
            return uinteger32(-1);

        uinteger32 ki = morton_keys[i];
        uinteger32 kj = morton_keys[i+1];

        return ki ^ kj;
    }

    GRACE_HOST_DEVICE uinteger64 operator()(
        const int i,
        const uinteger64* morton_keys,
        const size_t n_keys) const
    {
        if (i < 0 || i + 1 >= n_keys)
            return uinteger64(-1);

        uinteger64 ki = morton_keys[i];
        uinteger64 kj = morton_keys[i+1];

        return ki ^ kj;

    }
};

// Euclidian distance metric.
template <typename PrimitiveIter, typename CentroidFunc>
struct DeltaEuclidean
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;

    GRACE_HOST_DEVICE DeltaEuclidean() : centroid(CentroidFunc()) {}

    GRACE_HOST_DEVICE float operator()(
        const int i,
        PrimitiveIter primitives,
        const size_t n_primitives) const
    {
        if (i < 0 || i + 1 >= n_primitives) {
#ifdef __CUDA_ARCH__
            return CUDART_INF_F;
#else
            return std::numeric_limits<float>::infinity();
#endif
        }

        TPrimitive pi = primitives[i];
        TPrimitive pj = primitives[i+1];

        float3 ci = centroid(pi);
        float3 cj = centroid(pj);

        return (pi.x - pj.x) * (pi.x - pj.x)
               + (pi.y - pj.y) * (pi.y - pj.y)
               + (pi.z - pj.z) * (pi.z - pj.z);
    }

private:
    const CentroidFunc centroid;
};

// Surface area 'distance' metric.
template <typename PrimitiveIter, typename AABBFunc>
struct DeltaSurfaceArea
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;

    GRACE_HOST_DEVICE DeltaSurfaceArea() : AABB(AABBFunc()) {}

    GRACE_HOST_DEVICE float operator()(
        const int i,
        PrimitiveIter primitives,
        const size_t n_primitives) const
    {
        if (i < 0 || i + 1 >= n_primitives) {
#ifdef __CUDA_ARCH__
            return CUDART_INF_F;
#else
            return std::numeric_limits<float>::infinity();
#endif
        }

        TPrimitive pi = primitives[i];
        TPrimitive pj = primitives[i+1];

        float3 boti, topi, botj, topj;
        AABB(pi, &boti, &topi);
        AABB(pj, &botj, &topj);

        float L_x = max(topi.x, topj.x) - min(boti.x, botj.x);
        float L_y = max(topi.y, topj.y) - min(boti.y, botj.y);
        float L_z = max(topi.z, topj.z) - min(boti.z, botj.z);

        float SA = (L_x * L_y) + (L_x * L_z) + (L_y * L_z);

        return SA;
    }

private:
    const AABBFunc AABB;
};

struct AABBSphere
{
    template <typename Real4>
    GRACE_HOST_DEVICE void operator()(
        Real4 sphere,
        float3* bot,
        float3* top) const
    {
        bot->x = sphere.x - sphere.w;
        top->x = sphere.x + sphere.w;

        bot->y = sphere.y - sphere.w;
        top->y = sphere.y + sphere.w;

        bot->z = sphere.z - sphere.w;
        top->z = sphere.z + sphere.w;
    }
};

struct CentroidSphere
{
    template <typename Real4>
    GRACE_HOST_DEVICE float3 operator()(Real4 sphere) const
    {
        return make_float3(sphere.x, sphere.y, sphere.z);
    }
};

} // namespace grace
