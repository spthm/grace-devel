#pragma once

#include "grace/cuda/detail/device/intrinsics.cuh"
#include "grace/generic/intersect.h"

#include "grace/types.h"

namespace grace {

GRACE_DEVICE int AABBs_hit(
    const float3 invd, const float3 origin, const float len,
    const float4 AABB_L,
    const float4 AABB_R,
    const float4 AABB_LR)
{
    float bx_L = (AABB_L.x - origin.x) * invd.x;
    float tx_L = (AABB_L.y - origin.x) * invd.x;
    float by_L = (AABB_L.z - origin.y) * invd.y;
    float ty_L = (AABB_L.w - origin.y) * invd.y;
    float bz_L = (AABB_LR.x - origin.z) * invd.z;
    float tz_L = (AABB_LR.y - origin.z) * invd.z;

    float bx_R = (AABB_R.x - origin.x) * invd.x;
    float tx_R = (AABB_R.y - origin.x) * invd.x;
    float by_R = (AABB_R.z - origin.y) * invd.y;
    float ty_R = (AABB_R.w - origin.y) * invd.y;
    float bz_R = (AABB_LR.z - origin.z) * invd.z;
    float tz_R = (AABB_LR.w - origin.z) * invd.z;

    float tmin_L = maxf_vmaxf( fmin(bx_L, tx_L), fmin(by_L, ty_L),
                               maxf_vminf(bz_L, tz_L, 0) );
    float tmax_L = minf_vminf( fmax(bx_L, tx_L), fmax(by_L, ty_L),
                               minf_vmaxf(bz_L, tz_L, len) );
    float tmin_R = maxf_vmaxf( fmin(bx_R, tx_R), fmin(by_R, ty_R),
                               maxf_vminf(bz_R, tz_R, 0) );
    float tmax_R = minf_vminf( fmax(bx_R, tx_R), fmax(by_R, ty_R),
                               minf_vmaxf(bz_R, tz_R, len) );

    return (int)(tmax_R >= tmin_R) + 2*((int)(tmax_L >= tmin_L));
}

} // namespace grace
