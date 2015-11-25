#include "intersection.cuh"

#include <gmpxx.h>

double impact_parameter2(const grace::Ray& ray, const float4& sphere)
{
    mpq_class dx = mpq_class(ray.dx);
    mpq_class dy = mpq_class(ray.dy);
    mpq_class dz = mpq_class(ray.dz);

    // P = vec(ray.origin) - vec(sphere.origin)
    mpq_class px = mpq_class(ray.ox) - mpq_class(sphere.x);
    mpq_class py = mpq_class(ray.oy) - mpq_class(sphere.y);
    mpq_class pz = mpq_class(ray.oz) - mpq_class(sphere.z);

    // dot(P, ray.dir); ray.dir is normalized.
    mpq_class d = px * dx + py * dy + pz * dz;

    mpq_class bx = px - (d * dx);
    mpq_class by = py - (d * dy);
    mpq_class bz = pz - (d * dz);

    mpq_class b2 = bx * bx + by * by + bz * bz;

    return b2.get_d();
}
