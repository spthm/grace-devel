#pragma once

#include "interval.h"
#include "quadratic.h"

#include "ray.h"

#include <gmpxx.h>

/* Compute the quadratic equation for ray-sphere intersection.
 * Real roots of this equation are points of intersection along the ray,
 * which is parameterized by the scalar t,
 *    vec(ray) = vec(ray.origin) + t * vec(ray.dir)
 * where vec(ray.dir) is normalized.
 */
template <typename T>
VertexQuadratic<T> ray_sphere_equation(
    const grace::Ray& ray,
    const float4& sphere)
{
    // dot(vec(ray.dir), vec(ray.dir))
    // This *should* be == 1, but in practice there will be some error.
    // A reference ray-sphere intersection implementation should work with the
    // exact input data, rather than make assumptions about that data. Hence, we
    // do not assume dot(vec(ray.dir), vec(ray.dir)) == 1.
    T a = static_cast<T>(ray.dx) * ray.dx +
          static_cast<T>(ray.dy) * ray.dy +
          static_cast<T>(ray.dz) * ray.dz;

    // Sphere center -> ray origin vector.
    // vec(p) = vec(ray.origin) - vec(sphere.origin)
    T px, py, pz;
    px = static_cast<T>(ray.ox) - sphere.x;
    py = static_cast<T>(ray.oy) - sphere.y;
    pz = static_cast<T>(ray.oz) - sphere.z;

    // Quadratic equation b coefficient.
    // 2 * dot(vec(ray.dir), vec(p))
    T b = 2 * (ray.dx * px + ray.dy * py + ray.dz * pz);

    // Quadratic equation c coefficient.
    // dot(vec(p), vec(p)) - (sphere.radius)^2
    T c = px * px + py * py + pz * pz - (static_cast<T>(sphere.w) * sphere.w);

    return VertexQuadratic<T>(a, b, c);
}

/* Test for ray-sphere intersection.
 * The precision of all internal calculations is specified by the template
 * parameter.
 * ray_sphere_intersection<mpq_class> offers the most accurate results possible
 * given the input types (of the ray and sphere's internal variables).
 */
template <typename T>
bool ray_sphere_intersection(
    const grace::Ray& ray,
    const float4& sphere)
{
    Interval<T> ray_interv
        = Interval<T>(static_cast<T>(0), static_cast<T>(ray.length));

    VertexQuadratic<T> eqn = ray_sphere_equation<T>(ray, sphere);

    return eqn.root_exists(ray_interv);
}

/* Return the square of the ray-sphere impact parameter.
 * Internally, all calculations use GNU MP rationals; b^2 is computed exactly.
 * The returned double is the double conversion of the exact b^2.
 */
double impact_parameter2(const grace::Ray& ray, const float4& sphere);
