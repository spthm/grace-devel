#pragma once

#include "grace/cuda/kernels/gen_rays.cuh"

#include "grace/types.h"
#include "grace/ray.h"

#include <thrust/device_vector.h>

namespace grace {

// Generates isotropically distributed rays, emanating from a point.
// ox, oy and oz are the co-ordinates of the origin for all rays.
// length is the length of all rays.
// seed may optionally be specified as a seed to the underlying random number
//      generator. Using the same seed on the same device with the same origin
//      will always generate identical rays. Identical seeds across multiple
//      devices are not guaranteed to generate identical rays.
template <typename Real>
GRACE_HOST void uniform_random_rays(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const unsigned long long seed = 1234)
{
    detail::uniform_random_rays(d_rays_ptr, N_rays, ox, oy, oz, length, seed);
}

template <typename Real>
GRACE_HOST void uniform_random_rays(
    thrust::device_vector<Ray>& d_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const unsigned long long seed = 1234)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    const size_t N_rays = d_rays.size();

    uniform_random_rays(d_rays_ptr, N_rays, ox, oy, oz, length, seed);
}

// Similar to uniform_random_rays, except rays are confined to a single octant.
// The octant is specified as 'XYZ', where X, Y and Z are one of P or M.
// A P indicates that the X, Y or Z direction component of all rays should be
// positive.
// An M indicates that the X, Y or Z direction component of all rays should be
// negative.
// E.g. PPP: all rays are in the +ve x, +ve y and +ve z octant;
//      MPM: all rays are in the -ve x, +ve y and -ve z octant.
// Octants are defined in grace/types.h.
template <typename Real>
GRACE_HOST void uniform_random_rays_single_octant(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const enum Octants octant = PPP,
    const unsigned long long seed = 1234)
{
    detail::uniform_random_rays_single_octant(d_rays_ptr, N_rays, ox, oy, oz,
                                              length, octant, seed);
}

template <typename Real>
GRACE_HOST void uniform_random_rays_single_octant(
    thrust::device_vector<Ray>& d_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const enum Octants octant = PPP,
    const unsigned long long seed = 1234)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    const size_t N_rays = d_rays.size();

    uniform_random_rays_single_octant(d_rays_ptr, N_rays, ox, oy, oz, length,
                                      octant, seed);

}

// Generates rays emanating from a single point to N other points.
// ox, oy and oz are the co-ordinates of the origin for all rays.
// d_points_ptr points to an array of Real3-like (.x, .y and .z members)
//              co-ordinates, specifying the location of each ray's end point.
template <typename Real, typename Real3>
GRACE_HOST void one_to_many_rays(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real3* const d_points_ptr)
{
    detail::one_to_many_rays(d_rays_ptr, N_rays, ox, oy, oz, d_points_ptr);
}

// If d_rays.size() < d_points.size(), d_rays will be resized.
template <typename Real, typename Real3>
GRACE_HOST void one_to_many_rays(
    thrust::device_vector<Ray>& d_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const thrust::device_vector<Real3>& d_points)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    const Real3* const d_points_ptr = thrust::raw_pointer_cast(d_points.data());
    const size_t N_rays = d_points.size();
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }

    one_to_many_rays(d_rays_ptr, N_rays, ox, oy, oz, d_points_ptr);
}

// width and height: the dimensions of the grid of rays to generate
// base: a point at one corner of the ray-grid plane
// w: a vector defining the width of the plane, beginning from base
//    n = width rays are generated along w from base, for m = height rows
// h: a vector defining the height of the plane, beginning from base
//    m = height rays are generated along h from base, for n = width columns
// length: the length of all rays
//
// d_rays will be resized only if its current size is too small to contain
// width * height rays; it will never be reduced in size.
//
// A grid of width * height cells, covering an area |w| * |h|, is produced.
// base is located at one corner of this grid.
// The direction of all rays is normalize(cross(w, h)).
//
// One ray is generated for each N = width * height cells, with each ray's
// origin at a random location within its cell. Each ray-cell pair is unique.
// This introduces some randomness to the distribution while attempting to
// maintain relatively uniform sampling of the plane.
// For the same combination of inputs, the same rays will always be generated on
// the same hardware. This does not otherwise hold.
// For different rays within the same plane, provide a different seed parameter.
//
// No guarantees are made for the ordering of rays, other than that the order
// should be efficient for tracing.
//
// For rays originating at z = 10, travelling in the -z direction, and covering
// an area |w| * |h| = 5 * 6 = 30:
// base = (5, 0, 10)
// w = (-5, 0, 0)
// h = (0, 6, 0)
// direction = normalize(cross(w, h)) = normalize((0, 0, -30)) = (0, 0, -1)
template <typename Real, typename Real3>
GRACE_HOST void plane_parallel_random_rays(
    Ray* const d_rays_ptr,
    const int width,
    const int height,
    const Real3 base,
    const Real3 w,
    const Real3 h,
    const Real length,
    const unsigned long long seed = 1234)
{
    detail::plane_parallel_random_rays(d_rays_ptr, width, height, base, w, h,
                                       length, seed);
}

template <typename Real, typename Real3>
GRACE_HOST void plane_parallel_random_rays(
    thrust::device_vector<Ray>& d_rays,
    const int width,
    const int height,
    const Real3 base,
    const Real3 w,
    const Real3 h,
    const Real length,
    const unsigned long long seed = 1234)
{
    const size_t N_rays = static_cast<size_t>(width) * height;
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());

    plane_parallel_random_rays(d_rays_ptr, width, height, base, w, h, length,
                               seed);
}

// Similar to plane_parallel_random_rays, except ray origins are fixed at the
// cell centres for a given grid. Useful for emulating an orthogonal projection
// camera.
// Rays are ordered such that they increase along the direction of w first, then
// h.
//
// d_rays will be resized only if its current size is too small to contain
// width * height rays; it will never be reduced in size.
//
// width and height: the dimensions of the grid of rays to generate
// base: a point at one corner of the ray-grid plane
// w: a vector defining the width of the plane, beginning from base
//    n = width rays are generated along w from base, for m = height rows
// h: a vector defining the height of the plane, beginning from base
//    m = height rays are generated along h from base, for n = width columns
// length: the length of all rays
//
// A grid of width * height cells, covering an area |w| * |h|, is produced.
// The centre of each cell in this grid defines a ray origin. base is located at
// one corner of this grid. Hence, no ray will have origin == base.
// The direction of all rays is normalize(cross(w, h)).
//
// For rays originating at z = 10, travelling in the -z direction, and covering
// an area |w| * |h| = 5 * 6 = 30:
// base = (5, 0, 10)
// w = (-5, 0, 0)
// h = (0, 6, 0)
// direction = normalize(cross(w, h)) = normalize((0, 0, -30)) = (0, 0, -1)
template <typename Real, typename Real3>
GRACE_HOST void orthogonal_projection_rays(
    Ray* const d_rays_ptr,
    const int width,
    const int height,
    const Real3 base,
    const Real3 w,
    const Real3 h,
    const Real length)
{
    detail::orthogonal_projection_rays(d_rays_ptr, width, height, base, w, h,
                                       length);
}

template <typename Real, typename Real3>
GRACE_HOST void orthogonal_projection_rays(
    thrust::device_vector<Ray>& d_rays,
    const int width,
    const int height,
    const Real3 base,
    const Real3 w,
    const Real3 h,
    const Real length)
{
    const size_t N_rays = static_cast<size_t>(width) * height;
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());

    orthogonal_projection_rays(d_rays_ptr, width, height, base, w, h, length);
}

} // namespace grace
