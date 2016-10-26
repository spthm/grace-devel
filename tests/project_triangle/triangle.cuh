#pragma once

#include "helper/read_ply.hpp"
#include "helper/vector_math.cuh"

#include "grace/ray.h"

#define AABB_EPSILON 0.000001f
#define TRIANGLE_EPSILON 1E-14f

struct Triangle
{
    float3 v;
    float3 e1;
    float3 e2;

    // Must be constructable on the device.
    // thrust::sort[_by_key] requires a default constructor.
    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(float3 vertex, float3 edge1, float3 edge2)
        : v(vertex), e1(edge1), e2(edge2) {}
    __host__ __device__ Triangle(const PLYTriangle& tri)
        : v(tri.v1), e1(tri.v2 - tri.v1), e2(tri.v3 - tri.v1) {}
};

/* Moeller-Trumbore intersection algorithm.
 * "Fast, Minimum Storage Ray/Triangle Intersection", Moeller & Trumbore.
 * Journal of Graphics Tools, 1997.
 *
 * For best performance, this function should _not_ be compiled with -dc and
 * then linked to; the function definition should be available to the compiler
 * for all invocations. (It requires many temporary variables, which should be
 * visible to nvcc when compiling any kernels which call it so register usage
 * can be optimized.)
 *
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
inline __device__ float intersect(
    const grace::Ray& ray, const Triangle& tri, float* const t)
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

struct TriangleAABB
{
    // Must be callable from the device.
    __host__ __device__
    void operator()(const Triangle& tri, float3* bot, float3* top) const;
};

struct TriangleCentroid
{
    // Must be callable from the device.
    __host__ __device__
    float3 operator()(const Triangle& tri) const
    {
        // For triangle with vertices V0, V1 and V2, the centroid is located at
        // (1/3) * (V0 + V1 + V2).
        return tri.v + (1. / 3.) * (tri.e1 + tri.e2);
    }
};
