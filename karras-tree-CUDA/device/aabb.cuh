#pragma once

#include <iterator>

namespace grace {

namespace AABB {

GRACE_DEVICE float3 AABB_centroid(const float3 bot, const float3 top)
{
    float3 centre;
    centre.x = (static_cast<double>(bot.x) + top.x) / 2.;
    centre.y = (static_cast<double>(bot.y) + top.y) / 2.;
    centre.z = (static_cast<double>(bot.z) + top.z) / 2.;

    return centre;
}

} // namespace AABB

} // namespace grace
