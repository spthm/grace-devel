#include "triangle.cuh"

/* Moeller-Trumbore intersection algorithm.
 * "Fast, Minimum Storage Ray/Triangle Intersection", Moeller & Trumbore.
 * Journal of Graphics Tools, 1997.

 * The co-ordinate system undergoes a convenient transformation,
 * (x, y, z) -> (t, u, v), described below.
 *
 * The triangle is first translated to the ray origin, then rotated such
 * that the new axis t is aligned with the ray direction. Finally, it
 * is scaled such that the lengths of its sides along both new axes u and v
 * are equal to one.
 *
 * The ray-triangle intersection point is found in the (t, u, v) system.
 * If t > ray length, the intersection is beyond the ray terminus.
 * If any of u > 0, v > 0, u + v < 1 do not hold, the ray misses the
 * triangle (the point where the ray intersects the plane of the triangle
 * is not within the triangle's edges).
 *
 * The implementation here is that presented by Moeller and, Trumbore, with
 * an additional check on t at the end.
 */
__device__ float intersect(const grace::Ray& ray, const Triangle& tri,
                           float* const t)
{
    float3 dir = make_float3(ray.dx, ray.dy, ray.dz);
    float3 O = make_float3(ray.ox, ray.oy, ray.oz);

    float3 P = cross_product(dir, tri.e2);
    float det = dot_product(tri.e1, P);
    // If true, the ray lies in - or close to - the plane of the triangle.
    // Do not cull back faces.
    // if (det > -TRIANGLE_EPSILON && det < TRIANGLE_EPSILON) return false;
    // Cull back faces.
    if (det < TRIANGLE_EPSILON) return false;

    float inv_det = 1. / det;
    float3 OV = O - tri.v;
    float u = dot_product(OV, P) * inv_det;
    // If true, the ray intersects the plane of triangle outside the
    // triangle.
    if (u < 0.f || u > 1.f) return false;

    float3 Q = cross_product(OV, tri.e1);
    float v = dot_product(dir, Q) * inv_det;
    // If true, the ray intersects the plane of triangle outside the
    // triangle.
    if (v < 0.f || u + v > 1.f) return false;

    *t = dot_product(tri.e2, Q) * inv_det;

    return true;
}

__host__ __device__ void TriangleAABB::operator()(
    const Triangle& tri, float3* bot, float3* top) const
{
    float3 v0 = tri.v;
    float3 v1 = v0 + tri.e1;
    float3 v2 = v0 + tri.e2;

    bot->x = min(v0.x, min(v1.x, v2.x));
    bot->y = min(v0.y, min(v1.y, v2.y));
    bot->z = min(v0.z, min(v1.z, v2.z));

    top->x = max(v0.x, max(v1.x, v2.x));
    top->y = max(v0.y, max(v1.y, v2.y));
    top->z = max(v0.z, max(v1.z, v2.z));

    // Some PLY files contain triangles which are zero-sized in one or more
    // dimensions, but GRACE is not robust to zero-sized AABBs.
    if (bot->x == top->x) {
        float scale = abs(bot->x);
        bot->x -= AABB_EPSILON * scale;
        top->x += AABB_EPSILON * scale;
    }
    if (bot->y == top->y) {
        float scale = abs(bot->y);
        bot->y -= AABB_EPSILON * scale;
        top->y += AABB_EPSILON * scale;
    }
    if (bot->z == top->z) {
        float scale = abs(bot->z);
        bot->z -= AABB_EPSILON * scale;
        top->z += AABB_EPSILON * scale;
    }
}
