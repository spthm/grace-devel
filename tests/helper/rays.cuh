#pragma once

#include "ray.h"

#include "device/morton.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

// Generates orthogonal rays in the +z direction, covering all points in the
// range (mins.x/y/z - maxs.w) to (maxs.x/y/z + maxs.w).
// min/maxs.w may be safely set to zero to avoid edge effects.
void orthogonal_rays_z(const size_t N_side,
                       const float4 mins, const float4 maxs,
                       thrust::device_vector<grace::Ray>& d_rays,
                       float* area = NULL)
{
    const size_t N_rays = N_side * N_side;
    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<grace::uinteger32> h_keys(N_rays);

    // Rays emitted from box side (x, y, mins.z - maxs.w) and of length
    // (maxs.z + maxs.w) - (mins.z - maxs.w).
    // Since we generally want to fully include all
    // particles, the ray (ox, oy) limits are set by:
    //   [mins.x - maxs.w, maxs.x + maxs.w] and
    //   [mins.y - maxs.w, maxs.y + maxs.w].
    // Rays at the edges are likely to have no hits!
    float2 spacer;
    float3 O;
    float4 span;

    span.x = 2 * maxs.w + maxs.x - mins.x;
    span.y = 2 * maxs.w + maxs.y - mins.y;
    span.z = 2 * maxs.w + maxs.z - mins.z;

    spacer.x = span.x / (N_side - 1);
    spacer.y = span.y / (N_side - 1);

    O.x = mins.x - maxs.w;
    O.z = mins.z - maxs.w;

    for (int i = 0; i < N_side; ++i, O.x += spacer.x)
    {
        O.y = mins.y - maxs.w;

        for (int j = 0; j < N_side; ++j, O.y += spacer.y)
        {
            h_rays[i * N_side + j].dx = 0.0f;
            h_rays[i * N_side + j].dy = 0.0f;
            h_rays[i * N_side + j].dz = 1.0f;

            h_rays[i * N_side + j].ox = O.x;
            h_rays[i * N_side + j].oy = O.y;
            h_rays[i * N_side + j].oz = O.z;

            h_rays[i * N_side + j].length = span.z;

            // Since all rays are PPP, base key on origin instead.
            // Floats must be in (0, 1) for morton_key().
            h_keys[i * N_side + j]
                = grace::morton::morton_key((O.x - mins.x) / span.x,
                                            (O.y - mins.y) / span.y,
                                            0.0f);
        }
    }
    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());

    d_rays = h_rays;
    if (area != NULL) *area = spacer.x * spacer.y;
}
