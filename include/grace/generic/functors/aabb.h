#pragma once

#include <thrust/functional.h>

#include <iterator>

namespace grace {

namespace AABB {

// Finds the centroid of an AABB.
// The performance hit of using doubles here is not particularly high.
GRACE_HOST_DEVICE float3 AABB_centroid(const float3 bot, const float3 top)
{
    float3 centre;
    centre.x = (static_cast<double>(bot.x) + top.x) / 2.;
    centre.y = (static_cast<double>(bot.y) + top.y) / 2.;
    centre.z = (static_cast<double>(bot.z) + top.z) / 2.;

    return centre;
}

} // namespace AABB

// Converts from TPrimitive to a float3 (primitive centroid).
template <typename TPrimitive, typename AABBFunc>
GRACE_HOST_DEVICE float3 primitive_centroid(
    const TPrimitive primitive,
    const AABBFunc AABB)
{
    float3 bot, top;
    AABB(primitive, &bot, &top);
    return AABB::AABB_centroid(bot, top);
}

// Functor to convert from TPrimitive to a float3 (primitive centroid), taking
// the primitive's centroid to be the centroid of the primitive's AABB.
// AABBFunc must be declared __host__ __device__, a Thrust requirement.
template <typename TPrimitive, typename AABBFunc>
struct PrimitiveCentroid : public thrust::unary_function<TPrimitive, float3>
{
public:
    GRACE_HOST_DEVICE PrimitiveCentroid() : AABB(AABBFunc()) {}

    GRACE_HOST_DEVICE float3 operator()(TPrimitive primitive)
    {
        return primitive_centroid(primitive, AABBFunc());
    }

private:
    const AABBFunc AABB;
};

} // namespace grace
