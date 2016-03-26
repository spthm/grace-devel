#pragma once

#include "ray.h"

#include "device/morton.cuh"
#include "kernels/gen_rays.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

// Generates orthogonal rays in the +z direction, covering all ray-grid cells in
// the range (mins.x/y/z - maxs.w) to (maxs.x/y/z + maxs.w).
// The very edge points are not included, because ray origins are the centres
// of cells in the ray-grid.
// min/maxs.w may be safely set to zero to avoid edge effects.
void orthogonal_rays_z(const size_t N_side,
                       const float4 mins, const float4 maxs,
                       thrust::device_vector<grace::Ray>& d_rays,
                       float* area = NULL)
{
    // Rays emitted from box side (x, y, mins.z - maxs.w) and of length
    // (maxs.z + maxs.w) - (mins.z - maxs.w).
    // Since we generally want to fully include all particles, the ray (ox, oy)
    // limits are set by:
    //   [mins.x - maxs.w, maxs.x + maxs.w] and
    //   [mins.y - maxs.w, maxs.y + maxs.w].
    // Rays at the edges are likely to have no hits!

    // maxs.w ~ maximum SPH radius. Offset x, y and z on all sides by this value
    // to avoid clipping particle volumes.
    const float span_x = maxs.x - mins.x + 2 * maxs.w;
    const float span_y = maxs.y - mins.y + 2 * maxs.w;
    const float span_z = maxs.z - mins.z + 2 * maxs.w;

    const float3 base = make_float3(mins.x - maxs.w,
                                    mins.y - maxs.w,
                                    mins.z - maxs.w);
    const float3 w = make_float3(span_x, 0.f, 0.f);
    const float3 h = make_float3(0.f, span_y, 0.f);

    grace::orthogonal_projection_rays(d_rays, N_side, N_side, base, w, h,
                                      span_z);

    if (area != NULL) {
        const float cell_x = span_x / N_side;
        const float cell_y = span_y / N_side;
        *area = cell_x * cell_y;
    }
}
