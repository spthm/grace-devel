#pragma once

#include "triangle.cuh"

#include "grace/aabb.h"
#include "grace/ray.h"
#include "grace/vector.h"
#include "grace/cuda/nodes.h"
#include "grace/generic/boundedptr.h"

#include <thrust/device_vector.h>

struct RayData_tri
{
    int hit_idx;
    float t_min;
};

struct PrimaryRayResult
{
    int idx;
    float t_min;
};

struct ShadowRayResult
{
    int idx;
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
        ray_data.hit_idx = tri_idx;
    }
};

struct RayEntry_tri
{
    __device__ void operator()(const int /*ray_idx*/, const grace::Ray& ray,
                               RayData_tri& ray_data,
                               const grace::BoundedPtr<char> /*smem_ptr*/) const
    {
        ray_data.hit_idx = -1;
        ray_data.t_min = ray.end * (1.f + AABB_EPSILON);
    }
};

struct PrimaryRayExit_tri
{
private:
    PrimaryRayResult* const store;

public:
    PrimaryRayExit_tri(PrimaryRayResult* const store) : store(store) {}

    __device__ void operator()(const int ray_idx, const grace::Ray& ray,
                               RayData_tri& ray_data,
                               const grace::BoundedPtr<char> /*smem_ptr*/) const
    {
        store[ray_idx].idx = ray_data.hit_idx;
        store[ray_idx].t_min = ray_data.t_min;
    }
};

struct ShadowRayExit_tri
{
private:
    ShadowRayResult* const store;

public:
    ShadowRayExit_tri(ShadowRayResult* const store) : store(store) {}

    __device__ void operator()(const int ray_idx, const grace::Ray& ray,
                               RayData_tri& ray_data,
                               const grace::BoundedPtr<char> /*smem_ptr*/) const
    {
        store[ray_idx].idx = ray_data.hit_idx;
    }
};

void generate_shadow_rays(
    const int light_index,
    const thrust::device_vector<grace::Vector<3, float> >& d_lights_pos,
    const thrust::device_vector<grace::Ray>& d_primary_rays,
    const thrust::device_vector<PrimaryRayResult>& d_primary_results,
    thrust::device_vector<grace::Ray>& d_shadow_rays);

void trace_primary_rays(
    const thrust::device_vector<grace::Ray>& d_primary_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    thrust::device_vector<PrimaryRayResult>& d_primary_results);

void trace_shadow_rays(
    const thrust::device_vector<grace::Ray>& d_shadow_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    thrust::device_ptr<ShadowRayResult> d_shadow_results_ptr);
