#pragma once

#include "triangle.cuh"

#include "grace/aabb.h"
#include "grace/ray.h"
#include "grace/vector.h"
#include "grace/cuda/bvh.cuh"
#include "grace/generic/boundedptr.h"

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
    __device__ bool operator()(
        const grace::Ray& ray, const Triangle& tri,
        RayData_tri& ray_data, const int /*lane*/,
        const grace::BoundedPtr<char> /*smem_ptr*/) const
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
    __device__ void operator()(const int /*ray_idx*/, const grace::Ray&,
                               RayData_tri& ray_data, const int tri_idx,
                               const Triangle&, const int /*lane*/,
                               const grace::BoundedPtr<char> /*smem_ptr*/) const
    {
        ray_data.data = tri_idx;
    }
};

struct RayEntry_tri
{
    __device__ void operator()(const int /*ray_idx*/, const grace::Ray& ray,
                               RayData_tri& ray_data,
                               const grace::BoundedPtr<char> /*smem_ptr*/) const
    {
        ray_data.data = -1;
        ray_data.t_min = ray.end * (1.f + AABB_EPSILON);
    }
};

void setup_cameras(
    const grace::AABB<float> aabb,
    const float FOVy_degrees,
    const int resolution_x, const int resolution_y,
    std::vector<grace::Vector<3, float> >& camera_positions,
    grace::Vector<3, float>* look_at,
    grace::Vector<3, float>* view_up,
    float* FOVy_radians, float* ray_length);

void trace_closest_tri(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::CudaBvh& d_bvh,
    thrust::device_vector<int>& d_closest_tri_idx);
