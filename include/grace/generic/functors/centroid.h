#pragma once

#include "grace/types.h"
#include "grace/generic/functors/aabb.h"

// CUDA
#include <vector_functions.h>

#include <functional>

namespace grace {

// Functor to convert from TPrimitive to a float3 (primitive centroid), taking
// the primitive's centroid to be the centroid of the primitive's AABB.
// AABBFunc must be declared __host__ __device__, a Thrust requirement.
template <typename TPrimitive, typename AABBFunc>
struct PrimitiveCentroid : public std::unary_function<TPrimitive, float3>
{
public:
    GRACE_HOST_DEVICE PrimitiveCentroid() : AABB(AABBFunc()) {}

    GRACE_HOST_DEVICE float3 operator()(TPrimitive primitive)
    {
        float3 bot, top;
        AABB(primitive, &bot, &top);
        return detail::AABB_centroid(bot, top);
    }

private:
    const AABBFunc AABB;
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
