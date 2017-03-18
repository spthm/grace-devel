#pragma once

#include "grace/cuda/generate_rays.cuh"

#include "grace/ray.h"
#include "grace/vector.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

template <typename T>
GRACE_HOST grace::Vector<3, T> box_center(const grace::Sphere<T> mins,
                                          const grace::Sphere<T> maxs)
{
    T two = 2;
    grace::Vector<3, T> center = (mins.center() + maxs.center()) / two;
    return center;
}

template <typename T>
GRACE_HOST grace::Vector<3, T> box_span(const grace::Sphere<T> mins,
                                        const grace::Sphere<T> maxs)
{
    // maxs.r == padding beyond bounds (e.g. maximum SPH radius).
    // Offset x, y and z on all sides by this value to avoid clipping (e.g. of
    // SPH particle volumes).
    T two = 2;
    grace::Vector<3, T> span = maxs.center() - mins.center() + two * maxs.r;
    return span;
}

template <typename T>
GRACE_HOST float per_ray_area(const grace::Vector<3, T> span,
                              const size_t N_side)
{
    const float cell_x = span.x / N_side;
    const float cell_y = span.y / N_side;
    return cell_x * cell_y;
}

// Generates orthogonal rays in the -z direction.
// Rays are emitted from box side (x, y, maxs.z + maxs.r) and are of length >=
// (maxs.z + maxs.r) - (mins.z - maxs.r).
// Since we generally want to fully include all primitives, the ray (ox, oy)
// limits are set by
//   [mins.x - maxs.r, maxs.x + maxs.r] and
//   [mins.y - maxs.r, maxs.y + maxs.r],
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
// min/maxs.r may be safely set to zero.
template <typename T>
GRACE_HOST void orthogonal_rays_z(
    const size_t N_side,
    const grace::Sphere<T> mins, const grace::Sphere<T> maxs,
    thrust::device_vector<grace::Ray>& d_rays,
    float* area = NULL)
{
    grace::Vector<3, T> center = box_center(mins, maxs);
    grace::Vector<3, T> span = box_span(mins, maxs);
    // Maintain square aspect ratio, but contain everything in bounds.
    if (span.x > span.y) { span.y = span.x; }
    else if (span.y > span.x) { span.x = span.y; }

    if (area != NULL) {
        *area = per_ray_area(span, N_side);
    }

    grace::Vector<3, T> camera_position(center.x, center.y, span.z);
    grace::Vector<3, T> look_at = center;
    grace::Vector<3, T> view_direction(0.f, 0.f, -1.f);
    grace::Vector<3, T> view_up(0.f, 1.f, 0.f);
    float length = 2 * span.z;

    grace::orthographic_projection_rays(camera_position, look_at, view_up,
                                        span.y, length, N_side, N_side, d_rays);
}

// Generates semi-randomized plane-parallel rays in the +z direction, covering
// all ray-grid cells in range (mins.x/y/z - maxs.r) to (maxs.x/y/z + maxs.r).
// Ray origins are randomized with each ray's cell, and ray order is not
// specified (i.e. may not be useful for image generation).
template <typename T, typename StateT>
void plane_parallel_rays_z(const size_t N_side,
                           const grace::Sphere<T> mins,
                           const grace::Sphere<T> maxs,
                           grace::RngStates<StateT>& rng,
                           thrust::device_vector<grace::Ray>& d_rays,
                           float* area = NULL)
{
    grace::Vector<3, T> span = box_span(mins, maxs);

    grace::Vector<3, T> base(mins.x - maxs.r, // start in top-left for images
                             maxs.y + maxs.r, // start in top-left for images
                             maxs.z + maxs.r);
    // Ray direction is cross(w, h) = -z.
    grace::Vector<3, T> w(span.x, 0.f, 0.f);  // start in top-left for images
    grace::Vector<3, T> h(0.f, -span.y, 0.f); // start in top-left for images

    float length = 2 * span.z;

    if (area != NULL) {
        *area = per_ray_area(span, N_side);
    }

    grace::plane_parallel_random_rays(base, w, h, length, N_side, N_side, rng,
                                      d_rays);
}
