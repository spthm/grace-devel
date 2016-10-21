#pragma once

#include "triangle.cuh"

#include "helper/vector_math.cuh"

#include "grace/cuda/functors/trace.cuh"
#include "grace/cuda/kernels/bintree_trace.cuh"

#include <thrust/device_vector.h>

#define DIFFUSE_COEF 1.0f
#define AMBIENT_BKG 0.05f

__global__ void flat_shade_kernel(
    const grace::Ray* rays,
    const Triangle* triangles,
    const RayData_tri* raydata,
    const size_t N_rays,
    const float3* const lights_pos,
    const size_t N_lights,
    float* brightness)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_rays;
         tid += gridDim.x)
    {
        grace::Ray ray = rays[tid];
        int tri_idx = raydata[tid].data;
        float t_min = raydata[tid].t_min;

        double b = 0;
        if (tri_idx != -1)
        {
            Triangle tri = triangles[tri_idx];

            // Ray-triangle intersection point.
            float3 O = make_float3(ray.ox, ray.oy, ray.oz);
            float3 D = make_float3(ray.dx, ray.dy, ray.dz);
            float3 point = O + t_min * D;

            // Triangle normal.
            float3 normal = normalize(cross_product(tri.e1, tri.e2));

            for (int li = 0; li < N_lights; ++li)
            {
                float3 light_pos = lights_pos[li];
                // Point to light source vector.
                float3 L = normalize(light_pos - point);

                b += DIFFUSE_COEF * max(0.0, dot_product(L, normal));
            }

            b += AMBIENT_BKG;
        }

        brightness[tid] = b;
    }
}

void flat_shade_tris(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const thrust::device_vector<RayData_tri>& d_raydata_out,
    const thrust::device_vector<float3>& d_lights_pos,
    thrust::device_vector<float>& d_brightness)
{
    const size_t N_rays = d_rays.size();
    const int NT = 128;
    const int blocks = min((int)((N_rays + NT - 1) / NT), grace::MAX_BLOCKS);

    flat_shade_kernel<<<NT, blocks>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_tris.data()),
        thrust::raw_pointer_cast(d_raydata_out.data()),
        N_rays,
        thrust::raw_pointer_cast(d_lights_pos.data()),
        d_lights_pos.size(),
        thrust::raw_pointer_cast(d_brightness.data())
    );
}

void setup_lights(
    const float3 bots, const float3 tops,
    thrust::device_vector<float3>& d_lights_pos)
{
    float3 centre = make_float3((bots.x + tops.x) / 2.,
                                (bots.y + tops.y) / 2.,
                                (bots.z + tops.z) / 2.);
    float max_span = max(tops.x - bots.x,
                         max(tops.y - bots.y, tops.z - bots.z));

    // Above
    d_lights_pos.push_back(
        make_float3(centre.x, tops.y + max_span, tops.z + max_span)
    );
    // Left
    // d_lights_pos.push_back(
    //     make_float3(bots.x - max_span, centre.y, tops.z + max_span)
    // );
}

void trace_flat_shade_tri(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    const thrust::device_vector<float3>& d_lights_pos,
    thrust::device_vector<float>& d_brightness)
{
    thrust::device_vector<RayData_tri> d_raydata_out(d_rays.size());
    grace::trace_texref<RayData_tri>(
        d_rays,
        d_tris,
        d_tree,
        0,
        grace::Init_null(),
        RayIntersect_tri(),
        OnHit_tri(),
        RayEntry_tri(),
        // This copies RayData_tri to the provided array for each ray.
        RayExit_tri(thrust::raw_pointer_cast(d_raydata_out.data()))
    );

    flat_shade_tris(d_rays, d_tris, d_raydata_out, d_lights_pos, d_brightness);
}
