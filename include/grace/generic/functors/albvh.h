#pragma once

#include "grace/aabb.h"
#include "grace/types.h"
#include "grace/vector.h"

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

        Vector<3, float> ci = centroid(pi);
        Vector<3, float> cj = centroid(pj);

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

    GRACE_HOST_DEVICE DeltaSurfaceArea() : aabb_op(AABBFunc()) {}

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

        AABB<float> aabbi, aabbj;
        aabb_op(pi, &aabbi);
        aabb_op(pj, &aabbj);

        float L_x = max(aabbi.max.x, aabbj.max.x) - min(aabbi.min.x, aabbj.min.x);
        float L_y = max(aabbi.max.y, aabbj.max.y) - min(aabbi.min.y, aabbj.min.y);
        float L_z = max(aabbi.max.z, aabbj.max.z) - min(aabbi.min.z, aabbj.min.z);

        float SA = (L_x * L_y) + (L_x * L_z) + (L_y * L_z);

        return SA;
    }

private:
    const AABBFunc aabb_op;
};

} // namespace grace
