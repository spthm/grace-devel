#pragma once

#include "helper/vector_math.h"

#define TRIANGLE_EPSILON 0.000001

struct Triangle
{
    float3 v;
    float3 e1;
    float3 e2;

    GRACE_HOST_DEVICE Triangle(float3 vertex, float3 edge1, float3 edge2)
        : v(vertex), e1(edge1), e2(edge2) {}
};

struct RayData_tri
{
    float t_min;
}

struct RayIntersect_tri
{
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
    GRACE_DEVICE bool operator()(
        const grace::Ray& ray, const Triangle& tri,
        const RayData_tri& ray_data, const int /*lane*/,
        const grace::gpu::BoundIter<char> /*sm_iter*/) const
    {
        float3 dir = make_float3(ray.dx, ray.dy, ray.dz);
        float3 O = make_float3(ray.oz, ray.oy, ray.oz);

        float3 P = cross_product(dir, tri.e2);
        float det = dot_product(tri.e1, P);
        // If true, the ray lies in - or close to - the plane of the triangle.
        if (det > -TRIANGLE_EPSILON && det < TRIANGLE_EPSILON) return false;

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

        float t = dot_product(tri.e2, Q) * inv_det;
        // If true, the intersection is too far along the ray, or before the ray
        // origin.
        if (t > ray_data.tmin || t > ray.length || t < TRIANGLE_EPSILON) return false;

        ray_data.t_min = t;
        return true;
    }
};

struct RayEntry_tri
{
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const grace::Ray& ray,
                                 const RayData_tri& ray_data,
                                 const grace::gpu::BoundIter<char> /*smem_iter*/)
    {
        ray_data.tmin = ray.length;
    }
}

struct RayExit_tri
{
    GRACE_DEVICE void operator()(const int /*ray_idx*/, const grace::Ray&,
                                 const RayData_tru&,
                                 const grace::gpu::BoundIter<char> /*smem_iter*/)
}

struct TriangleAABB
{
    void operator()(const Triangle& tri, float3* bot, float3* top) const
    {
        float3 v0 = tri.v;
        float3 v1 = v0 + tri.e1;
        float3 v2 = v0 + tri.e2;

        *bot.x = min(v0.x, min(v1.x, v2.x));
        *bot.y = min(v0.y, min(v1.y, v2.y));
        *bot.z = min(v0.z, min(v1.z, v2.z));

        *top.x = max(v0.x, max(v1.x, v2.x));
        *top.y = max(v0.y, max(v1.y, v2.y));
        *top.z = max(v0.z, max(v1.z, v2.z));
    }
};

struct TriangleCentroid
{
    float3 operator()(const Triangle& tri)
    {
        // For triangle with vertices V0, V1 and V2, the centroid is located at
        // (1/3) * (V0 + V1 + V2).
        return tri.v + (1. / 3.) * (tri.e1 + tri.e2);
    }
};
