#pragma once

#include "grace/cuda/detail/device/intrinsics.cuh"
#include "grace/generic/intersect.h"

#include "grace/aabb.h"
#include "grace/types.h"
#include "grace/vector.h"

namespace grace {

GRACE_DEVICE int AABBs_hit(
    const Vector3f invd, const Vector3f origin,
    const float start, const float end,
    const AABBf AABB_L,
    const AABBf AABB_R)
{
    Vector3f b_L = (AABB_L.min - origin) * invd;
    Vector3f t_L = (AABB_L.max - origin) * invd;

    Vector3f b_R = (AABB_R.min - origin) * invd;
    Vector3f t_R = (AABB_R.max - origin) * invd;

    float tmin_L = maxf_vmaxf( fmin(b_L.x, t_L.x), fmin(b_L.y, t_L.y),
                               maxf_vminf(b_L.z, t_L.z, start) );
    float tmax_L = minf_vminf( fmax(b_L.x, t_L.x), fmax(b_L.y, t_L.y),
                               minf_vmaxf(b_L.z, t_L.z, end) );
    float tmin_R = maxf_vmaxf( fmin(b_R.x, t_R.x), fmin(b_R.y, t_R.y),
                               maxf_vminf(b_R.z, t_R.z, start) );
    float tmax_R = minf_vminf( fmax(b_R.x, t_R.x), fmax(b_R.y, t_R.y),
                               minf_vmaxf(b_R.z, t_R.z, end) );

    return (int)(tmax_R >= tmin_R) + 2*((int)(tmax_L >= tmin_L));
}

} // namespace grace
