#pragma once

#include "intrinsics.cuh"
#include "util.cuh"

#include "../types.h"

namespace grace {

namespace gpu {

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

// Computations all happen with the precision of the type Real.
template <typename Real4, typename Real>
GRACE_DEVICE bool sphere_hit(
    const Ray& ray,
    const Real4& sphere,
    Real& b2,
    Real& dot_p)
{
    Real px = sphere.x - ray.ox;
    Real py = sphere.y - ray.oy;
    Real pz = sphere.z - ray.oz;

    // Already normalized.
    Real rx = ray.dx;
    Real ry = ray.dy;
    Real rz = ray.dz;

    // Distance to intersection.
    dot_p = px * rx + py * ry + pz * rz;
    // dot_p = fma(px, rx, fma(py, ry, pz * rz));

    // Impact parameter.
    // negations mean -fma(a, b, -c) is not a clear win. Let the compiler decide.
    Real bx = px - dot_p * rx;
    Real by = py - dot_p * ry;
    Real bz = pz - dot_p * rz;
    b2 = bx * bx + by * by + bz * bz;
    // b2 = fma(bx, bx, fma(by, by, bz * bz));

    if (b2 >= sphere.w * sphere.w)
        return false;

    // If dot_p < 0, the ray origin must be inside the sphere for an
    // intersection. We treat this edge-case as a miss.
    if (dot_p < 0.0f)
        return false;

    // If dot_p > ray length, the ray terminus must be inside the sphere for
    // an intersection. We treat this edge-case as a miss.
    if (dot_p >= ray.length)
        return false;

    // Otherwise, assume we have a hit.  This counts the following partial
    // intersections as hits:
    //     i) Ray starts inside sphere, before point of closest approach.
    //    ii) Ray ends inside sphere, beyond point of closest approach.
    return true;
}

} // namespace gpu

} // namespace grace
