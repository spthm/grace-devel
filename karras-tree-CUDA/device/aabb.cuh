#pragma once

#include <iterator>

namespace grace {

namespace AABB {

// Finds the centroid of an AABB.
GRACE_HOST_DEVICE float3 AABB_centroid(const float3 bot, const float3 top)
{
    float3 centre;
    centre.x = (static_cast<double>(bot.x) + top.x) / 2.;
    centre.y = (static_cast<double>(bot.y) + top.y) / 2.;
    centre.z = (static_cast<double>(bot.z) + top.z) / 2.;

    return centre;
}

// Converts from TPrimitive to a float3 (primitive centroid).
template <typename TPrimitive, typename AABBFunc>
GRACE_HOST_DEVICE float3 primitive_centroid(
    const TPrimitive primitive,
    const AABBFunc AABB)
{
    float3 bot, top;
    AABB(primitive, &bot, &top);
    return AABB_centroid(bot, top);
}

// Functor to convert from TPrimitive to a float3 (primitive centroid).
// AABBFunc must be declared __host__ __device__, a Thrust requirement.
template <typename TPrimitive, typename AABBFunc>
struct CentroidFunc : public thrust::unary_function<TPrimitive, float3>
{
private:
    AABBFunc AABB;
public:
    GRACE_HOST_DEVICE CentroidFunc() : AABB(AABBFunc()) {}

    GRACE_HOST_DEVICE float3 operator()(TPrimitive primitive)
    {
        return AABB::primitive_centroid(primitive, AABBFunc());
    }
};

} // namespace AABB

} // namespace grace
