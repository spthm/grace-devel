#include "tris_trace.cuh"

#include "helper/vector_math.cuh"

#include "grace/cuda/kernel_config.h"
#include "grace/cuda/functors/trace.cuh"
#include "grace/cuda/kernels/bintree_trace.cuh"

static __global__ void shadow_rays_kernel(
    const grace::Ray* const primary_rays,
    const size_t N_rays,
    const PrimaryRayResult* const primary_results,
    const int light_index,
    const float3* lights_pos,
    grace::Ray* const shadow_rays)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_rays;
         tid += gridDim.x)
    {
        grace::Ray ray = primary_rays[tid];
        int tri_idx = primary_results[tid].idx;

        grace::Ray shadow_ray;
        if (tri_idx != -1)
        {
            float t_min = primary_results[tid].t_min;

            float3 O = make_float3(ray.ox, ray.oy, ray.oz);
            float3 D = make_float3(ray.dx, ray.dy, ray.dz);

            // Point needs to be moved off surface to prevent numerical artefacts.
            float3 point = O + fmaxf(0.f, (t_min - 1e-4f)) * D;
            float3 light_vector = lights_pos[light_index] - point;
            float light_distance = magnitude(light_vector);
            float3 L = light_vector / light_distance;

            shadow_ray.ox = point.x; shadow_ray.oy = point.y; shadow_ray.oz = point.z;
            shadow_ray.dx = L.x; shadow_ray.dy = L.y; shadow_ray.dz = L.z;
            shadow_ray.length = light_distance;
        }
        else
        {
            shadow_ray.ox = 0.f; shadow_ray.oy = 0.f; shadow_ray.oz = 0.f;
            shadow_ray.dx = 0.f; shadow_ray.dy = 0.f; shadow_ray.dz = 1.f;
            shadow_ray.length = -1.f;
        }

        shadow_rays[tid] = shadow_ray;
    }
}

void generate_shadow_rays(
    const int light_index,
    const thrust::device_vector<float3>& d_lights_pos,
    const thrust::device_vector<grace::Ray>& d_primary_rays,
    const thrust::device_vector<PrimaryRayResult>& d_primary_results,
    thrust::device_vector<grace::Ray>& d_shadow_rays)
{
    const int NT = 128;
    const int blocks = min((int)((d_primary_rays.size() + NT - 1) / NT), grace::MAX_BLOCKS);
    shadow_rays_kernel<<<NT, blocks>>>(
        thrust::raw_pointer_cast(d_primary_rays.data()),
        d_primary_rays.size(),
        thrust::raw_pointer_cast(d_primary_results.data()),
        light_index,
        thrust::raw_pointer_cast(d_lights_pos.data()),
        thrust::raw_pointer_cast(d_shadow_rays.data())
    );

}

void trace_primary_rays(
    const thrust::device_vector<grace::Ray>& d_primary_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    thrust::device_vector<PrimaryRayResult>& d_primary_results)
{
    grace::trace_texref<RayData_tri>(
            d_primary_rays,
            d_tris,
            d_tree,
            0,
            grace::Init_null(),
            RayIntersect_tri(),
            OnHit_tri(),
            RayEntry_tri(),
            // This copies RayData_tri to the provided array for each ray.
            PrimaryRayExit_tri(thrust::raw_pointer_cast(d_primary_results.data()))
        );
}

void trace_shadow_rays(
    const thrust::device_vector<grace::Ray>& d_shadow_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    thrust::device_ptr<ShadowRayResult> d_shadow_results_ptr)
{
    grace::trace_texref<RayData_tri>(
            d_shadow_rays,
            d_tris,
            d_tree,
            0,
            grace::Init_null(),
            RayIntersect_tri(),
            OnHit_tri(),
            RayEntry_tri(),
            ShadowRayExit_tri(thrust::raw_pointer_cast(d_shadow_results_ptr))
        );
}
