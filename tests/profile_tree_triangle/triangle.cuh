#pragma once

#include "helper/read_ply.hpp"

#include "grace/aabb.h"
#include "grace/vector.h"

#define AABB_EPSILON 0.000001f
#define TRIANGLE_EPSILON 1E-14f

struct Triangle
{
    grace::Vector<3, float> v;
    grace::Vector<3, float> e1;
    grace::Vector<3, float> e2;

    // Must be constructable on the device.
    // thrust::sort[_by_key] requires a default constructor.
    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(grace::Vector<3, float> vertex,
                                 grace::Vector<3, float> edge1,
                                 grace::Vector<3, float> edge2)
        : v(vertex), e1(edge1), e2(edge2) {}
    __host__ __device__ Triangle(const PLYTriangle& tri)
        : v(tri.v1), e1(tri.v2 - tri.v1), e2(tri.v3 - tri.v1) {}
};

struct TriangleAABB
{
    // Must be callable from the device.
    __host__ __device__
    void operator()(const Triangle& tri, grace::AABB<float>* aabb) const;
};

struct TriangleCentroid
{
    // Must be callable from the device.
    __host__ __device__
    grace::Vector<3, float> operator()(const Triangle& tri) const
    {
        // For triangle with vertices V0, V1 and V2, the centroid is located at
        // (1/3) * (V0 + V1 + V2).
        return tri.v + (float)(1. / 3.) * (tri.e1 + tri.e2);
    }
};
