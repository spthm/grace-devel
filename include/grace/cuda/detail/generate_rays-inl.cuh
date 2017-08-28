#pragma once

#include "grace/cuda/detail/kernels/generate_rays.cuh"
#include "grace/cuda/prngstates.cuh"
#include "grace/cuda/util/extrema.cuh"

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

template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays(
    const Vector<3, Real> origin,
    const Real length,
    const size_t N_rays,
    RngStates<StateT>& rng,
    Ray* const d_rays_ptr)
{
    detail::uniform_random_rays(origin, length, N_rays, rng.device_states(),
                                d_rays_ptr);
}

template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays(
    const Vector<3, Real> origin,
    const Real length,
    RngStates<StateT>& rng,
    thrust::device_vector<Ray>& d_rays)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    const size_t N_rays = d_rays.size();

    detail::uniform_random_rays(origin, length, N_rays, rng.device_states(),
                                d_rays_ptr);
}


template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays_single_octant(
    const Vector<3, Real> origin,
    const Real length,
    const size_t N_rays,
    RngStates<StateT>& rng,
    Ray* const d_rays_ptr,
    const enum Octants octant)
{
    detail::uniform_random_rays_single_octant(origin, length, octant, N_rays,
                                              rng.device_states(), d_rays_ptr);
}

template <typename Real, typename StateT>
GRACE_HOST void uniform_random_rays_single_octant(
    const Vector<3, Real> origin,
    const Real length,
    RngStates<StateT>& rng,
    thrust::device_vector<Ray>& d_rays,
    const enum Octants octant)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    const size_t N_rays = d_rays.size();

    detail::uniform_random_rays_single_octant(origin, length, octant, N_rays,
                                              rng.device_states(), d_rays_ptr);

}


template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays(
    const Vector<3, Real> origin,
    const PointType* const d_points_ptr,
    const size_t N_rays,
    Ray* const d_rays_ptr,
    const enum RaySortType sort_type)
{
    if (sort_type == NoSort) {
        detail::one_to_many_rays_nosort(origin, d_points_ptr, N_rays,
                                        d_rays_ptr);
    }
    else if (sort_type == DirectionSort) {
        detail::one_to_many_rays_dirsort(origin, d_points_ptr, N_rays,
                                         d_rays_ptr);
    }
    else if (sort_type == EndPointSort) {
        AABB<float> aabb;
        min_vec3(d_points_ptr, N_rays, &aabb.min);
        max_vec3(d_points_ptr, N_rays, &aabb.max);
        detail::one_to_many_rays_endsort(origin, d_points_ptr, N_rays, aabb,
                                         d_rays_ptr);
    }
    else {
        std::stringstream msg_stream;
        msg_stream << "Ray sort type not recognized";
        const std::string msg = msg_stream.str();

        throw std::invalid_argument(msg);
    }
}

template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays(
    const Vector<3, Real> origin,
    const thrust::device_vector<PointType>& d_points,
    thrust::device_vector<Ray>& d_rays,
    const enum RaySortType sort_type)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    const PointType* const d_points_ptr
        = thrust::raw_pointer_cast(d_points.data());
    const size_t N_rays = d_points.size();
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }

    one_to_many_rays(origin, d_points_ptr, N_rays, d_rays_ptr, sort_type);
}

template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays_endsort(
    const Vector<3, Real> origin,
    const PointType* const d_points_ptr,
    const size_t N_rays,
    const AABB<Real>& aabb,
    Ray* const d_rays_ptr)
{
    detail::one_to_many_rays_endsort(origin, d_points_ptr, N_rays, aabb,
                                     d_rays_ptr);
}

// If d_rays.size() < d_points.size(), d_rays will be resized.
template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays_endsort(
    const Vector<3, Real> origin,
    const thrust::device_vector<PointType>& d_points,
    const AABB<Real>& aabb,
    thrust::device_vector<Ray>& d_rays)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    const PointType* const d_points_ptr
        = thrust::raw_pointer_cast(d_points.data());
    const size_t N_rays = d_points.size();
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }

    detail::one_to_many_rays_endsort(origin, d_points_ptr, N_rays, aabb,
                                     d_rays_ptr);
}


template <typename Real, typename StateT>
GRACE_HOST void plane_parallel_random_rays(
    const Vector<3, Real>& base,
    const Vector<3, Real>& w,
    const Vector<3, Real>& h,
    const Real length,
    const int width,
    const int height,
    RngStates<StateT>& rng,
    Ray* const d_rays_ptr)
{
    detail::plane_parallel_random_rays(base, w, h, length, width, height,
                                       rng.device_states(), d_rays_ptr);
}

template <typename Real, typename StateT>
GRACE_HOST void plane_parallel_random_rays(
    const Vector<3, Real>& base,
    const Vector<3, Real>& w,
    const Vector<3, Real>& h,
    const Real length,
    const int width,
    const int height,
    RngStates<StateT>& rng,
    thrust::device_vector<Ray>& d_rays)
{
    const size_t N_rays = (size_t)width * height;
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());

    detail::plane_parallel_random_rays(base, w, h, length, width, height,
                                       rng.device_states(), d_rays_ptr);
}


template <typename Real>
GRACE_HOST void orthographic_projection_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real vertical_extent,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    Ray* const d_rays_ptr)
{
    detail::orthographic_projection_rays(camera_position, look_at, view_up,
                                         vertical_extent, length,
                                         resolution_x, resolution_y,
                                         d_rays_ptr);
}

template <typename Real>
GRACE_HOST void orthographic_projection_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real vertical_extent,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    thrust::device_vector<Ray>& d_rays)
{
    const size_t N_rays = (size_t)resolution_x * resolution_y;
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());

    detail::orthographic_projection_rays(camera_position, look_at, view_up,
                                         vertical_extent, length,
                                         resolution_x, resolution_y,
                                         d_rays_ptr);
}


template <typename Real>
GRACE_HOST void pinhole_camera_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real FOVy,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    Ray* const d_rays_ptr)
{
    detail::pinhole_camera_rays(camera_position, look_at, view_up, FOVy,
                                length, resolution_x, resolution_y,
                                d_rays_ptr);
}

template <typename Real>
GRACE_HOST void pinhole_camera_rays(
    const Vector<3, Real> camera_position,
    const Vector<3, Real> look_at,
    const Vector<3, Real> view_up,
    const Real FOVy,
    const Real length,
    const int resolution_x,
    const int resolution_y,
    thrust::device_vector<Ray>& d_rays)
{
    const size_t N_rays = (size_t)resolution_x * resolution_y;
    if (d_rays.size() < N_rays) {
        d_rays.resize(N_rays);
    }
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());

    detail::pinhole_camera_rays(camera_position, look_at, view_up, FOVy,
                                length, resolution_x, resolution_y,
                                d_rays_ptr);
}

} // namespace grace
