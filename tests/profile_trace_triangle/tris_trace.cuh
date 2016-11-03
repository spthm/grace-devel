#pragma once

#include "triangle.cuh"

#include "grace/ray.h"
#include "grace/cuda/nodes.h"
#include "grace/cuda/util/bound_iter.cuh"

#include <thrust/device_vector.h>

struct RayData_tri
{
    // We can make use of some of GRACE's built-in generic traversal functors
    // if our RayData type has a public .data member.
    int data;
    float t_min;
};

struct RayIntersect_tri
{
    /* This method definition, and the definition of the intersect() function it
     * calls, should be visible at compile-time for all device-side invocations
     * for best performance! If this method instead links to some external
     * __device__ function, performance is significantly reduced, because nvcc
     * cannot optimize the register usage here or within intersect() in the
     * context of the calling kernel.
     */
    // grace::gpu::BoundIter is not callable on the host.
    __device__ bool operator()(
        const grace::Ray& ray, const Triangle& tri,
        RayData_tri& ray_data, const int /*lane*/,
        const grace::gpu::BoundIter<char> /*sm_iter*/) const
    {
        float t;
        bool hit = false;
        if (intersect(ray, tri, &t))
        {
            // If false, the intersection is too far along the ray, or before
            // the ray origin.
            if (t <= ray_data.t_min && t >= TRIANGLE_EPSILON)
            {
                ray_data.t_min = t;
                hit = true;
            }
        }

        return hit;
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


void trace_closest_tri(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    thrust::device_vector<int>& d_closest_tri_idx);
