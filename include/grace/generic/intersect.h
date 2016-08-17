#pragma once

#include "grace/ray.h"
#include "grace/types.h"

namespace grace {

// Computations all happen with the precision of the type Real.
template <typename Real4, typename Real>
GRACE_HOST_DEVICE bool sphere_hit(
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

    // If dot_p < ray start, the ray origin must be inside the sphere for an
    // intersection. We treat this edge-case as a miss.
    if (dot_p < ray.start)
        return false;

    // If dot_p > ray end, the ray terminus must be inside the sphere for
    // an intersection. We treat this edge-case as a miss.
    if (dot_p >= ray.end)
        return false;

    // Otherwise, assume we have a hit.  This counts the following partial
    // intersections as hits:
    //     i) Ray starts inside sphere, before point of closest approach.
    //    ii) Ray ends inside sphere, beyond point of closest approach.
    return true;
}

} // namespace grace
