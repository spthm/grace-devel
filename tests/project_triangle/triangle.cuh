#pragma once

#include "helper/read_ply.hpp"
#include "helper/vector_math.cuh"

#include "grace/cuda/util/bound_iter.cuh"
#include "grace/cuda/nodes.h"

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

struct RayData_tri
{
    // We can make use of some of GRACE's built-in generic traversal functors
    // if our RayData type has a public .data member.
    int data;
    float t_min;
};

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
    // grace::gpu::BoundIter is not callable on the host.
    __device__ bool operator()(
        const grace::Ray& ray, const Triangle& tri,
        RayData_tri& ray_data, const int /*lane*/,
        const grace::gpu::BoundIter<char> /*sm_iter*/) const
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

        float t = dot_product(tri.e2, Q) * inv_det;
        // If true, the intersection is too far along the ray, or before the ray
        // origin.
        if (t > ray_data.t_min || t > ray.length || t < TRIANGLE_EPSILON) return false;

        ray_data.t_min = t;
        return true;
    }
};

struct OnHit_tri
{
    // grace::gpu::BoundIter is not callable on the host.
    __device__ void operator()(const int /*ray_idx*/, const grace::Ray&,
                               RayData_tri& ray_data, const int tri_idx,
                               const Triangle&, const int /*lane*/,
                               const grace::gpu::BoundIter<char> /*smem_iter*/) const
    {
        ray_data.data = tri_idx;
    }
};

struct RayEntry_tri
{
    // grace::gpu::BoundIter is not callable on the host.
    __device__ void operator()(const int /*ray_idx*/, const grace::Ray& ray,
                               RayData_tri& ray_data,
                               const grace::gpu::BoundIter<char> /*smem_iter*/) const
    {
        ray_data.data = -1;
        ray_data.t_min = ray.length * (1.f + AABB_EPSILON);
    }
};

struct RayExit_tri
{
private:
    RayData_tri* const store;

public:
    RayExit_tri(RayData_tri* const store) : store(store) {}

    // grace::gpu::BoundIter is not callable on the host.
    __device__ void operator()(const int ray_idx, const grace::Ray& ray,
                               RayData_tri& ray_data,
                               const grace::gpu::BoundIter<char> /*smem_iter*/) const
    {
        store[ray_idx] = ray_data;
    }
};

struct RayExit_shade_tri
{
private:
    const RayData_tri* const primary_raydata;
    const float* const tri_colours;
    float* const brightness;

public:
    RayExit_shade_tri(const RayData_tri* const primary_raydata,
                      float* const tri_colours,
                      float* const brightness) :
        primary_raydata(primary_raydata),
        tri_colours(tri_colours),
        brightness(brightness) {}

    // grace::gpu::BoundIter is not callable on the host.
    __device__ void operator()(const int ray_idx, const grace::Ray& ray,
                               RayData_tri& ray_data,
                               const grace::gpu::BoundIter<char> /*smem_iter*/) const
    {
        int tri_idx = primary_raydata[ray_idx].data;

        // If this light is not blocked (i.e. if there are no hits), and if the
        // corresponding primary ray did hit something, then shade according to
        // this light.
        if (ray_data.data == -1 && tri_idx != -1) {
            brightness[ray_idx] += tri_colours[tri_idx];
        }
    }
};

struct TriangleAABB
{
    // Must be callable from the device.
    __host__ __device__
    void operator()(const Triangle& tri, float3* bot, float3* top) const
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
