#pragma once

#include "helper/read_ply.hpp"
#include "helper/vector_math.cuh"

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
