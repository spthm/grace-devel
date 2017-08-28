#pragma once

#include "grace/config.h"
#include "grace/ray.h"
#include "grace/sphere.h"

namespace grace {

template <typename PrecisionType, typename T>
GRACE_HOST_DEVICE bool sphere_hit(
    const Ray& ray,
    const Sphere<T>& sphere,
    PrecisionType& b2,
    PrecisionType& dot_p)
{
    PrecisionType px = sphere.x - ray.ox;
    PrecisionType py = sphere.y - ray.oy;
    PrecisionType pz = sphere.z - ray.oz;

    // Already normalized.
    PrecisionType rx = ray.dx;
    PrecisionType ry = ray.dy;
    PrecisionType rz = ray.dz;

    // Distance to intersection.
    dot_p = px * rx + py * ry + pz * rz;
    // dot_p = fma(px, rx, fma(py, ry, pz * rz));

    // Impact parameter.
    // negations mean -fma(a, b, -c) is not a clear win. Let the compiler decide.
    PrecisionType bx = px - dot_p * rx;
    PrecisionType by = py - dot_p * ry;
    PrecisionType bz = pz - dot_p * rz;
    b2 = bx * bx + by * by + bz * bz;
    // b2 = fma(bx, bx, fma(by, by, bz * bz));

    if (b2 >= sphere.r * sphere.r)
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
