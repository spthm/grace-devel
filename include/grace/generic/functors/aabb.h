#pragma once

#include "grace/types.h"

#include <iterator>

namespace grace {

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

namespace detail {

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

} // namespace detail

} // namespace grace
