#pragma once

#include "triangle.cuh"

#include "helper/vector_math.cuh"

#include "grace/cuda/functors/trace.cuh"
#include "grace/cuda/kernels/bintree_trace.cuh"

#include <thrust/device_vector.h>

#define AMBIENT_BKG 0.05f

__global__ void shadow_rays_kernel(
    const grace::Ray* const primary_rays,
    const size_t N_rays,
    const RayData_tri* const primary_raydata,
    const int light_index,
    const float3* lights_pos,
    grace::Ray* const shadow_rays)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_rays;
         tid += gridDim.x)
    {
        grace::Ray ray = primary_rays[tid];
        int tri_idx = primary_raydata[tid].data;

        grace::Ray shadow_ray;
        if (tri_idx != -1)
        {
            float t_min = primary_raydata[tid].t_min;

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

void shadow_rays(
    const int light_index,
    const thrust::device_vector<float3>& d_lights_pos,
    const thrust::device_vector<grace::Ray>& d_primary_rays,
    const thrust::device_vector<RayData_tri>& d_primary_raydata,
    thrust::device_vector<grace::Ray>& d_shadow_rays)
{
    d_shadow_rays.resize(d_primary_rays.size());

    const int NT = 128;
    const int blocks = min((int)((d_primary_rays.size() + NT - 1) / NT), grace::MAX_BLOCKS);
    shadow_rays_kernel<<<NT, blocks>>>(
        thrust::raw_pointer_cast(d_primary_rays.data()),
        d_primary_rays.size(),
        thrust::raw_pointer_cast(d_primary_raydata.data()),
        light_index,
        thrust::raw_pointer_cast(d_lights_pos.data()),
        thrust::raw_pointer_cast(d_shadow_rays.data())
    );

}

__global__ void base_shade_kernel(
    const RayData_tri* const primary_raydata,
    const size_t N_rays,
    float* const brightness)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_rays;
         tid += gridDim.x)
    {
        int tri_idx = primary_raydata[tid].data;

        // Miss -> background colour; hit -> ambient colour.
        if (tri_idx == -1) brightness[tid] = 0.f;
        else brightness[tid] = AMBIENT_BKG;
    }
}

void base_shade(
    const thrust::device_vector<RayData_tri>& d_primary_raydata,
    thrust::device_vector<float>& d_brightness)
{
    const int NT = 128;
    const int blocks = min((int)((d_primary_raydata.size() + NT - 1) / NT), grace::MAX_BLOCKS);
    base_shade_kernel<<<NT, blocks>>>(
        thrust::raw_pointer_cast(d_primary_raydata.data()),
        d_primary_raydata.size(),
        thrust::raw_pointer_cast(d_brightness.data())
    );
}

__global__ void shade_tri_colours_kernel(
    const Triangle* triangles,
    const size_t N_tris,
    const float3* const lights_pos,
    const size_t N_lights,
    float* const shaded_colours)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_tris;
         tid += gridDim.x)
    {
        Triangle tri = triangles[tid];
        float3 normal = normalize(cross_product(tri.e1, tri.e2));

        for (int l = 0; l < N_lights; ++l)
        {
            float3 light_pos = lights_pos[l];
            float3 L = normalize(light_pos - tri.v);

            // The true value would vary with the point at which the ray
            // intersects the triangle. However, provided that
            // |L| >> |tri.e1|, |tri.e2| (i.e. the light is far away from the
            // triangle) the below is approximately correct.
            float shading = max(0.0, dot_product(L, normal));

            shaded_colours[l * N_tris + tid] = shading;
        }
    }
}

void shade_tri_colours(
    const thrust::device_vector<Triangle>& d_tris,
    const thrust::device_vector<float3>& d_lights_pos,
    thrust::device_vector<float>& d_tri_colours)
{
    d_tri_colours.resize(d_tris.size() * d_lights_pos.size());

    const int NT = 128;
    const int blocks = min((int)((d_tris.size() + NT - 1) / NT), grace::MAX_BLOCKS);
    shade_tri_colours_kernel<<<NT, blocks>>>(
        thrust::raw_pointer_cast(d_tris.data()),
        d_tris.size(),
        thrust::raw_pointer_cast(d_lights_pos.data()),
        d_lights_pos.size(),
        thrust::raw_pointer_cast(d_tri_colours.data())
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

void trace_shade_tri(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    const thrust::device_vector<float3>& d_lights_pos,
    thrust::device_vector<float>& d_brightness)
{
    thrust::device_vector<RayData_tri> d_primary_raydata(d_rays.size());
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
        RayExit_tri(thrust::raw_pointer_cast(d_primary_raydata.data()))
    );

    // Primary rays which miss -> background colour.
    // Primary rays which hit -> ambient colour/minimum brightness
    base_shade(d_primary_raydata, d_brightness);

    // Compute colour of each triangle for each light.
    thrust::device_vector<float> d_tri_colours;
    shade_tri_colours(d_tris, d_lights_pos, d_tri_colours);

    thrust::device_vector<grace::Ray> d_shadow_rays(d_rays.size());
    for (int i = 0; i < d_lights_pos.size(); ++i)
    {
        // Compute intersection point -> light position rays.
        shadow_rays(i, d_lights_pos, d_rays, d_primary_raydata, d_shadow_rays);
        // For all shadow rays which _do not_ hit anything, shade according
        // to that light/triangle combination.
        grace::trace_texref<RayData_tri>(
            d_shadow_rays,
            d_tris,
            d_tree,
            0,
            grace::Init_null(),
            RayIntersect_tri(),
            OnHit_tri(),
            RayEntry_tri(),
            // This adds to brightness the shading from d_tri_colours for all
            // triangles which are hit by a primary ray, and which are not
            // blocked (their shadow rays have no hits).
            RayExit_shade_tri(
                thrust::raw_pointer_cast(d_primary_raydata.data()),
                thrust::raw_pointer_cast(d_tri_colours.data()) + i * d_tris.size(),
                thrust::raw_pointer_cast(d_brightness.data())
            )
        );
    }
}
