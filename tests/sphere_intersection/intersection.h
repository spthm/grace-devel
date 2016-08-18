#pragma once

#include "interval.h"
#include "quadratic.h"

#include "grace/ray.h"
#include "grace/sphere.h"

#include <gmpxx.h>

/* Compute the quadratic equation for ray-sphere intersection.
 * Real roots of this equation are points of intersection along the ray,
 * which is parameterized by the scalar t,
 *    vec(ray) = vec(ray.origin) + t * vec(ray.dir)
 * where vec(ray.dir) is normalized.
 */
template <typename PrecisionType, typename T>
VertexQuadratic<PrecisionType> ray_sphere_equation(
    const grace::Ray& ray,
    const grace::Sphere<T>& sphere)
{
    // dot(vec(ray.dir), vec(ray.dir))
    // This *should* be == 1, but in practice there will be some error.
    // A reference ray-sphere intersection implementation should work with the
    // exact input data, rather than make assumptions about that data. Hence, we
    // do not assume dot(vec(ray.dir), vec(ray.dir)) == 1.
    PrecisionType a = static_cast<PrecisionType>(ray.dx) * ray.dx +
                      static_cast<PrecisionType>(ray.dy) * ray.dy +
                      static_cast<PrecisionType>(ray.dz) * ray.dz;

    // Sphere center -> ray origin vector.
    // vec(p) = vec(ray.origin) - vec(sphere.origin)
    PrecisionType px, py, pz;
    px = static_cast<PrecisionType>(ray.ox) - sphere.x;
    py = static_cast<PrecisionType>(ray.oy) - sphere.y;
    pz = static_cast<PrecisionType>(ray.oz) - sphere.z;

    // Quadratic equation b coefficient.
    // 2 * dot(vec(ray.dir), vec(p))
    PrecisionType b = 2 * (ray.dx * px + ray.dy * py + ray.dz * pz);

    // Quadratic equation c coefficient.
    // dot(vec(p), vec(p)) - (sphere.radius)^2
    PrecisionType c = px * px + py * py + pz * pz
                      - (static_cast<PrecisionType>(sphere.r) * sphere.r);

    return VertexQuadratic<PrecisionType>(a, b, c);
}

/* Test for ray-sphere intersection.
 * The precision of all internal calculations is specified by the template
 * parameter.
 * ray_sphere_intersection<mpq_class> offers the most accurate results possible
 * given the input types (of the ray and sphere's internal variables).
 */
template <typename PrecisionType, typename T>
bool ray_sphere_intersection(
    const grace::Ray& ray,
    const grace::Sphere<T>& sphere)
{
    Interval<PrecisionType> ray_interv
        = Interval<PrecisionType>(static_cast<PrecisionType>(0),
                                  static_cast<PrecisionType>(ray.end));

    VertexQuadratic<PrecisionType> eqn
        = ray_sphere_equation<PrecisionType>(ray, sphere);

    return eqn.root_exists(ray_interv);
}

/* Return the square of the ray-sphere impact parameter.
 * Internally, all calculations use GNU MP rationals; b^2 is computed exactly.
 * The returned double is the double conversion of the exact b^2.
 */
double impact_parameter2(const grace::Ray& ray, const grace::Sphere<double>& sphere);
float  impact_parameter2(const grace::Ray& ray, const grace::Sphere<float>& sphere);
