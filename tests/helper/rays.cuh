#pragma once

#include "grace/cuda/gen_rays.cuh"

#include "grace/ray.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

// Setup for rays emitted from box side (x, y, mins.z - maxs.w) and of length
// (maxs.z + maxs.w) - (mins.z - maxs.w).
// Since we generally want to fully include all particles, the ray (ox, oy)
// limits are set by
//   [mins.x - maxs.w, maxs.x + maxs.w] and
//   [mins.y - maxs.w, maxs.y + maxs.w],
// but rays at the edges are likely to have no hits!
void setup_plane_z(const size_t N_side,
                   const float4 mins, const float4 maxs,
                   float3* const base, float3* const w, float3* const h,
                   float* const length, float* const area)
{
    // maxs.w ~ maximum SPH radius. Offset x, y and z on all sides by this value
    // to avoid clipping particle volumes.
    const float span_x = maxs.x - mins.x + 2 * maxs.w;
    const float span_y = maxs.y - mins.y + 2 * maxs.w;
    const float span_z = maxs.z - mins.z + 2 * maxs.w;

    *base = make_float3(mins.x - maxs.w,
                        mins.y - maxs.w,
                        mins.z - maxs.w);
    *w = make_float3(span_x, 0.f, 0.f);
    *h = make_float3(0.f, span_y, 0.f);

    *length = span_z;

    if (area != NULL) {
        const float cell_x = span_x / N_side;
        const float cell_y = span_y / N_side;
        *area = cell_x * cell_y;
    }
}

// Generates orthogonal rays in the +z direction, covering all ray-grid cells in
// the range (mins.x/y/z - maxs.w) to (maxs.x/y/z + maxs.w).
// The very edge points are not included, because ray origins are the centres
// of cells in the ray-grid.
// Rays are ordered to increase along x first, then y, hence are suitable for
// image generation.
// min/maxs.w may be safely set to zero to avoid edge effects in images.
void orthogonal_rays_z(const size_t N_side,
                       const float4 mins, const float4 maxs,
                       thrust::device_vector<grace::Ray>& d_rays,
                       float* area = NULL)
{
    float3 base, w, h;
    float length;
    setup_plane_z(N_side, mins, maxs, &base, &w, &h, &length, area);

    grace::orthogonal_projection_rays(d_rays, N_side, N_side, base, w, h,
                                      length);
}

// Generates semi-randomized plane-parallel rays in the +z direction, covering
// all ray-grid cells in range (mins.x/y/z - maxs.w) to (maxs.x/y/z + maxs.w).
// Ray origins are randomized with each ray's cell, and ray order is not
// specified (i.e. may not be useful for image generation).
void plane_parallel_rays_z(const size_t N_side,
                           const float4 mins, const float4 maxs,
                           thrust::device_vector<grace::Ray>& d_rays,
                           float* area = NULL,
                           unsigned int seed = 1234)
{
    float3 base, w, h;
    float length;
    setup_plane_z(N_side, mins, maxs, &base, &w, &h, &length, area);

    grace::plane_parallel_random_rays(d_rays, N_side, N_side, base, w, h,
                                      length, seed);
}
