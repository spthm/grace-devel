#pragma once

#include "grace/cuda/prngstates.cuh"

#include "grace/aabb.h"
#include "grace/config.h"
#include "grace/ray.h"
#include "grace/types.h"
#include "grace/vector.h"

#include <thrust/device_vector.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace grace {

// Generates isotropically distributed rays, emanating from a point.
// origin specifies co-ordinates of the origin for all rays.
// length is the length of all rays.
template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays(
    const Vector<3, Real> origin,
    const Real length,
    const size_t N_rays,
    RngStates<StateT>& rng,
    Ray* const d_rays_ptr);

template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays(
    const Vector<3, Real> origin,
    const Real length,
    RngStates<StateT>& rng,
    thrust::device_vector<Ray>& d_rays);

// Similar to uniform_random_rays, except rays are confined to a single octant.
// The octant is specified as 'XYZ', where X, Y and Z are one of P or M.
// A P indicates that the X, Y or Z direction component of all rays should be
// positive.
// An M indicates that the X, Y or Z direction component of all rays should be
// negative.
// E.g. PPP: all rays are in the +ve x, +ve y and +ve z octant;
//      MPM: all rays are in the -ve x, +ve y and -ve z octant.
// Octants are defined in grace/config.h.
template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays_single_octant(
    const Vector<3, Real> origin,
    const Real length,
    const size_t N_rays,
    RngStates<StateT>& rng,
    Ray* const d_rays_ptr,
    const enum Octants octant = PPP);

template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays_single_octant(
    const Vector<3, Real> origin,
    const Real length,
    RngStates<StateT>& rng,
    thrust::device_vector<Ray>& d_rays,
    const enum Octants octant = PPP);

// Generates rays emanating from a single point to N other points.
// origin specifies co-ordinates of the origin for all rays.
// d_points_ptr points to an array of Real3-like (.x, .y and .z members)
//              co-ordinates, specifying the location of each ray's end point.
template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays(
    const Vector<3, Real> origin,
    const PointType* const d_points_ptr,
    const size_t N_rays,
    Ray* const d_rays_ptr,
    const enum RaySortType sort_type = DirectionSort);

// If d_rays.size() < d_points.size(), d_rays will be resized.
template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays(
    const Vector<3, Real> origin,
    const thrust::device_vector<PointType>& d_points,
    thrust::device_vector<Ray>& d_rays,
    const enum RaySortType sort_type = DirectionSort);

// Generates rays emanating from a single point to N other points.
// origin specifies co-ordinates of the origin for all rays.
// d_points_ptr points to an array of Real3-like (.x, .y and .z members)
//              co-ordinates, specifying the location of each ray's end point.
// When an endpoint sort is desired, and AABBs for the input points are already
// known, this saves re-computing them, which would occur if calling
// one_to_many_rays without providing AABBs.
template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays_endsort(
    const Vector<3, Real> origin,
    const PointType* const d_points_ptr,
    const size_t N_rays,
    const AABB<Real>& aabb,
    Ray* const d_rays_ptr);

// If d_rays.size() < d_points.size(), d_rays will be resized.
template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays_endsort(
    const Vector<3, Real> origin,
    const thrust::device_vector<PointType>& d_points,
    const AABB<Real>& aabb,
    thrust::device_vector<Ray>& d_rays);

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
template <typename Real, typename StateT>
GRACE_HOST void plane_parallel_random_rays(
    const Vector<3, Real>& base,
    const Vector<3, Real>& w,
    const Vector<3, Real>& h,
    const Real length,
    const int width,
    const int height,
    RngStates<StateT>& rng,
    Ray* const d_rays_ptr);

template <typename Real, typename StateT>
GRACE_HOST void plane_parallel_random_rays(
    const Vector<3, Real>& base,
    const Vector<3, Real>& w,
    const Vector<3, Real>& h,
    const Real length,
    const int width,
    const int height,
    RngStates<StateT>& rng,
    thrust::device_vector<Ray>& d_rays);

// Generates rays for an orthographic projection; this is the projection
// achieved from a perspective projection an infinite distance from the object,
// and with an infinite focal length.
//
// Rays emanate in a common direction parallel to (look_at - camera_position)
// (the length of the resultant vector is not used), and the view orientation is
// specified by the vector view_up. resolution_x rays are generated along the
// horizontal, and resolution_y along the vertical, for a total of
// N = resolution_x * resolution_y rays.
//
// Square pixels are always assumed, and the aspect ratio of the image is
// given by the ratio resolution_x / resolution_y.
//
// The total extent rays cover along the image vertical is specified by
// vertical_extent; the corresponding horizontal_extent is the vertical extent
// multiplied by the aspect ratio.
//
// The ray length, length, is fixed for all rays.
//
// The ray at index 0 corresponds to the top-left of the image plane, and the
// ray at index resolution_x * resolution_y - 1 corresponds to the bottom-right
// of the image plane. Rays are ordered such that they increase along the
// horizontal first, then the vertical.
//
// Note that view_up need only be approximate. The upward direction in the image
// plane is parallel to the vector
//     view_up - view_direction * (dot(view_direction, view_up)),
// where view_direction is parallel to the vector (look_at - camera_position).
// That is to say, the vertical in the image plane is the direction view_up
// with all components in the direction of view_direction removed.
template <typename Real>
GRACE_HOST void orthographic_projection_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real vertical_extent,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    Ray* const d_rays_ptr);

template <typename Real>
GRACE_HOST void orthographic_projection_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real vertical_extent,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    thrust::device_vector<Ray>& d_rays);

// Generates rays in a pinhole-camera model. Rays emanate from a common origin
// camera_position and, have a vertical field-of-view FOVy in the image plane.
// FOVy should be provided in radians.
//
// The centre of view is located at look_at, and the view orientation is
// specified by the vector view_up. resolution_x rays are generated along the
// horizontal, and resolution_y along the vertical, for a total of
// N = resolution_x * resolution_y rays. Unlike a true pinhole-camera, the
// resulting image is not vertically flipped.
//
// Square pixels are always assumed, and the aspect ratio of the image is
// given by the ratio resolution_x / resolution_y. The horizontal field-of-view,
// FOVx, can be computed via
//     tan(FOVx / 2) = aspect_ratio * tan(FOVy / 2)
//
// The ray length, length, is fixed for all rays; note therefore that the plane
// on which all rays terminate is a spherical surface.
//
// The ray at index 0 corresponds to the top-left of the image plane, and the
// ray at index resolution_x * resolution_y - 1 corresponds to the bottom-right
// of the image plane. Rays are ordered such that they increase along the
// horizontal first, then the vertical.
//
// Note that view_up need only be approximate. The upward direction in the image
// plane is parallel to the vector
//     view_up - view_direction * (dot(view_direction, view_up)),
// where view_direction is parallel to the vector (look_at - camera_position).
// That is to say, the vertical in the image plane is the direction view_up
// with all components in the direction of view_direction removed.
template <typename Real>
GRACE_HOST void pinhole_camera_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real FOVy,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    Ray* const d_rays_ptr);

template <typename Real>
GRACE_HOST void pinhole_camera_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real FOVy,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    thrust::device_vector<Ray>& d_rays);

} // namespace grace

#include "grace/cuda/detail/generate_rays-inl.cuh"
