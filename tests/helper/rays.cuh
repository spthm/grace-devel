#pragma once

#include "grace/cuda/generate_rays.cuh"

#include "grace/ray.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

template <typename T>
GRACE_HOST float3 box_center(const grace::Sphere<T> mins,
                             const grace::Sphere<T> maxs)
{
    float3 center = make_float3((mins.x + maxs.x) / 2.,
                                (mins.y + maxs.y) / 2.,
                                (mins.z + maxs.z) / 2.);
    return center;
}

template <typename T>
GRACE_HOST float3 box_span(const grace::Sphere<T> mins,
                           const grace::Sphere<T> maxs)
{
    // maxs.r == padding beyond bounds (e.g. maximum SPH radius).
    // Offset x, y and z on all sides by this value to avoid clipping (e.g. of
    // SPH particle volumes).
    float3 span = make_float3(maxs.x - mins.x + 2 * maxs.r,
                              maxs.y - mins.y + 2 * maxs.r,
                              maxs.z - mins.z + 2 * maxs.r);
    return span;
}

GRACE_HOST float per_ray_area(const float3 span, const size_t N_side)
{
    const float cell_x = span.x / N_side;
    const float cell_y = span.y / N_side;
    return cell_x * cell_y;
}

// Generates orthogonal rays in the -z direction.
// Rays are emitted from box side (x, y, maxs.z + maxs.w) and are of length >=
// (maxs.z + maxs.w) - (mins.z - maxs.w); maxs.w is a padding term.
// Since we generally want to fully include all primitives, the ray (ox, oy)
// limits are set by
//   [mins.x - maxs.w, maxs.x + maxs.w] and
//   [mins.y - maxs.w, maxs.y + maxs.w],
// but rays at the edges are likely to have no hits.
// A square aspect ratio is always assumed. The range covered by rays may be
// stretched in x or y ensure that all points within the provided bounds are
// visible, whilst maintaining this square aspect ratio. That is, if the spatial
// range to cover is not square, rays will extend above and below, or left and
// right, of the values in mins and maxs.
// The very edge points may not be included, because ray origins are the centres
// of cells in the ray-grid.
// Rays are ordered to increase along x first, then y, hence are suitable for
// image generation.
// min/maxs.w may be safely set to zero.
template <typename T>
GRACE_HOST void orthogonal_rays_z(
    const size_t N_side,
    const grace::Sphere<T> mins, const grace::Sphere<T> maxs,
    thrust::device_vector<grace::Ray>& d_rays,
    float* area = NULL)
{
    float3 center = box_center(mins, maxs);
    float3 span = box_span(mins, maxs);
    // Maintain square aspect ratio, but contain everything in bounds.
    if (span.x > span.y) { span.y = span.x; }
    else if (span.y > span.x) { span.x = span.y; }

    if (area != NULL) {
        *area = per_ray_area(span, N_side);
    }

    float3 camera_position = make_float3(center.x, center.y, span.z);
    float3 look_at = center;
    float3 view_direction = make_float3(0.f, 0.f, -1.f);
    float3 view_up = make_float3(0.f, 1.f, 0.f);
    float length = 2 * span.z;

    grace::orthographic_projection_rays(d_rays, N_side, N_side, camera_position,
                                        look_at, view_up, span.y, length);
}

// Generates semi-randomized plane-parallel rays in the +z direction, covering
// all ray-grid cells in range (mins.x/y/z - maxs.r) to (maxs.x/y/z + maxs.r).
// Ray origins are randomized with each ray's cell, and ray order is not
// specified (i.e. may not be useful for image generation).
template <typename T>
void plane_parallel_rays_z(const size_t N_side,
                           const grace::Sphere<T> mins,
                           const grace::Sphere<T> maxs,
                           thrust::device_vector<grace::Ray>& d_rays,
                           float* area = NULL,
                           unsigned int seed = 1234)
{
    float3 span = box_span(mins, maxs);

    float3 base = make_float3(mins.x - maxs.w, // start in top-left for images
                              maxs.y + maxs.w, // start in top-left for images
                              maxs.z + maxs.w);
    // Ray direction is cross(w, h) = -z.
    float3 w = make_float3(span.x, 0.f, 0.f);  // start in top-left for images
    float3 h = make_float3(0.f, -span.y, 0.f); // start in top-left for images

    float length = 2 * span.z;

    if (area != NULL) {
        *area = per_ray_area(span, N_side);
    }

    grace::plane_parallel_random_rays(d_rays, N_side, N_side, base, w, h,
                                      length, seed);
}
