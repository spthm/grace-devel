#pragma once

#include "grace/cuda/gen_rays.cuh"

#include "grace/ray.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

// Setup for rays emitted from box side (x, y, maxs.z + maxs.w) and of length
// (maxs.z + maxs.w) - (mins.z - maxs.w).
// Since we generally want to fully include all primitives, the ray (ox, oy)
// limits are set by
//   [mins.x - maxs.w, maxs.x + maxs.w] and
//   [mins.y - maxs.w, maxs.y + maxs.w],
// but rays at the edges are likely to have no hits.
// Ray origins may extend above/below or left/right of the above ranges if said
// range does not represent a square area, to avoid stretching the image in x or
// y.
GRACE_HOST void setup_plane_z(
    const size_t N_side,
    const float4 mins, const float4 maxs,
    float3* const base, float3* const w, float3* const h,
    float* const length, float* const area)
{
    // maxs.w == padding beyond bounds (e.g. maximum SPH radius).
    // Offset x, y and z on all sides by this value to avoid clipping (e.g. of
    // SPH particle volumes).
    float span_x = maxs.x - mins.x + 2 * maxs.w;
    float span_y = maxs.y - mins.y + 2 * maxs.w;
    float span_z = maxs.z - mins.z + 2 * maxs.w;

    *base = make_float3(mins.x - maxs.w, // start in top-left for images
                        maxs.y + maxs.w, // start in top-left for images
                        maxs.z + maxs.w);
    // Ray direction is cross(w, h) = -z.
    *w = make_float3(span_x, 0.f, 0.f);  // start in top-left for images
    *h = make_float3(0.f, -span_y, 0.f); // start in top-left for images

    *length = span_z;

    if (span_x > span_y)
    {
        // Make span_y == span_x, modify h and base accordingly.
        base->y += (-span_y + span_x) / 2.;
        h->y = -span_x;
        span_y = span_x;
    }
    else if (span_y > span_x)
    {
        // Make span_x == span_y, modify w and base accordingly.
        base->x += (span_x - span_y) / 2.;
        w->x = span_y;
        span_x = span_y;
    }

    if (area != NULL) {
        const float cell_x = span_x / N_side;
        const float cell_y = span_y / N_side;
        *area = cell_x * cell_y;
    }
}

// Generates orthogonal rays in the -z direction, covering all ray-grid cells in
// the range (mins.x/y/z - maxs.w) to (maxs.x/y/z + maxs.w).
// The range covered by rays may be stretched in x or y to maintain the aspect
// ratio implied by the values of mins and maxs (that is, if the spatial range
// to cover is not square, rays will extend above and below, or left and right,
// of the values in mins and maxs).
// The very edge points may not be included, because ray origins are the centres
// of cells in the ray-grid.
// Rays are ordered to increase along x first, then y, hence are suitable for
// image generation.
// min/maxs.w may be safely set to zero.
GRACE_HOST void orthogonal_rays_z(
    const size_t N_side,
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
GRACE_HOST void plane_parallel_rays_z(
    const size_t N_side,
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
