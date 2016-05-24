#pragma once

#include "grace/cuda/functors/trace.cuh"
#include "grace/cuda/kernels/bintree_trace.cuh"
#include "grace/generic/meta.h"
#include "grace/generic/raydata.h"
#include "grace/error.h"
#include "grace/types.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace grace {

//-----------------------------------------------------------------------------
// SPH Kernel line integral lookup table, parameterized by normalized impact
// parameter.
//-----------------------------------------------------------------------------

const static int N_table = 51;

template <typename Real>
struct KernelIntegrals
{
    const static Real table[N_table];

};

template <typename Real>
    const Real KernelIntegrals<Real>::table[N_table] = {
    1.90986019771937, 1.90563449910964, 1.89304415940934, 1.87230928086763,
    1.84374947679902, 1.80776276033034, 1.76481079856299, 1.71540816859939,
    1.66011373131439, 1.59952322363667, 1.53426266082279, 1.46498233888091,
    1.39235130929287, 1.31705223652377, 1.23977618317103, 1.16121278415369,
    1.08201943664419, 1.00288866679720, 0.924475767210246, 0.847415371038733,
    0.772316688105931, 0.699736940377312, 0.630211918937167, 0.564194562399538,
    0.502076205853037, 0.444144023534733, 0.390518196140658, 0.341148855945766,
    0.295941946237307, 0.254782896476983, 0.217538645099225, 0.184059547649710,
    0.154181189781890, 0.127726122453554, 0.104505535066266,
    8.432088120445191E-002, 6.696547102921641E-002, 5.222604427168923E-002,
    3.988433820097490E-002, 2.971866601747601E-002, 2.150552303075515E-002,
    1.502124104014533E-002, 1.004371608622562E-002, 6.354242122978656E-003,
    3.739494884706115E-003, 1.993729589156428E-003, 9.212900163813992E-004,
    3.395908945333921E-004, 8.287326418242995E-005, 7.387919939044624E-006,
    0.000000000000000E+000
};

const static KernelIntegrals<double> lookup = {};


//-----------------------------------------------------------------------------
// C-like convenience wrappers for common forms of the tracing kernel.
//-----------------------------------------------------------------------------

template <typename Real4>
GRACE_HOST void trace_hitcounts_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<int>& d_hit_counts)
{
    // Defines only RayData.data, of type int.
    typedef RayData_datum<int> RayData;

    trace_texref<RayData>(
        d_rays,
        d_spheres,
        d_tree,
        0,
        Init_null(),
        Intersect_sphere_bool(),
        OnHit_increment(),
        RayEntry_null(),
        RayExit_to_array<int>(
            thrust::raw_pointer_cast(d_hit_counts.data()))
    );
}

template <typename Real4, typename Real>
GRACE_HOST void trace_cumulative_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<Real>& d_cumulated)
{
    // TODO: Change it such that this is passed in, rather than copying it on
    // each call.
    const size_t sm_table_size = sizeof(double) * N_table;
    const double* p_table = &(lookup.table[0]);
    thrust::device_vector<double> d_lookup(p_table, p_table + N_table);

    typedef RayData_sphere<Real, Real> RayData;
    trace_texref<RayData>(
        d_rays,
        d_spheres,
        d_tree,
        sm_table_size,
        InitGlobalToSmem<double>(
            thrust::raw_pointer_cast(d_lookup.data()),
            N_table),
        Intersect_sphere_b2dist(),
        OnHit_sphere_cumulate(N_table),
        RayEntry_null(),
        RayExit_to_array<Real>(
            thrust::raw_pointer_cast(d_cumulated.data()))
    );
}

template <typename Real4, typename IndexType, typename Real>
GRACE_HOST void trace_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require ray offsets to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    thrust::device_vector<Real>& d_hit_integrals,
    thrust::device_vector<Real>& d_hit_distances)
{
    const size_t n_rays = d_rays.size();

    // Initially, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts_sph(d_rays, d_spheres, d_tree, d_ray_offsets);
    int last_ray_hitcount = d_ray_offsets[n_rays - 1];

    // Allocate output array from total per-ray hit counts, and calculate
    // individual ray offsets into this array:
    //
    // hits = [3, 0, 4, 1]
    // exclusive_scan:
    //    => offsets = [0, 3, 3, 7]
    // total_hits = hits[3] + offsets[3] = 1 + 7 = 8
    thrust::exclusive_scan(d_ray_offsets.begin(), d_ray_offsets.end(),
                           d_ray_offsets.begin());
    int total_hits = d_ray_offsets[n_rays - 1] + last_ray_hitcount;

    d_hit_integrals.resize(total_hits);
    d_hit_indices.resize(total_hits);
    d_hit_distances.resize(total_hits);

    // TODO: Change it such that this is passed in, rather than copying it on
    // each call.
    const size_t sm_table_size = sizeof(double) * N_table;
    const double* p_table = &(lookup.table[0]);
    thrust::device_vector<double> d_lookup(p_table, p_table + N_table);

    typedef RayData_sphere<int, Real> RayData;
    trace_texref<RayData>(
        d_rays,
        d_spheres,
        d_tree,
        sm_table_size,
        InitGlobalToSmem<double>(
            thrust::raw_pointer_cast(d_lookup.data()),
            N_table),
        Intersect_sphere_b2dist(),
        OnHit_sphere_individual<IndexType, Real>(
            thrust::raw_pointer_cast(d_hit_indices.data()),
            thrust::raw_pointer_cast(d_hit_integrals.data()),
            thrust::raw_pointer_cast(d_hit_distances.data()),
            N_table),
        RayEntry_from_array<int>(
            thrust::raw_pointer_cast(d_ray_offsets.data())),
        RayExit_null()
    );
}

template <typename Real4, typename IndexType, typename Real>
GRACE_HOST void trace_with_sentinels_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require this to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    const int index_sentinel,
    thrust::device_vector<Real>& d_hit_integrals,
    const Real integral_sentinel,
    thrust::device_vector<Real>& d_hit_distances,
    const Real distance_sentinel)
{
    const size_t n_rays = d_rays.size();

    // Initially, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts(d_rays, d_ray_offsets, d_tree, d_spheres);
    int last_ray_hitcount = d_ray_offsets[n_rays-1];

    // Allocate output array from total per-ray hit counts, and calculate
    // individual ray offsets into this array:
    //
    // hits = [3, 0, 4, 1]
    // exclusive_scan:
    //     => offsets = [0, 3, 3, 7]
    thrust::exclusive_scan(d_ray_offsets.begin(), d_ray_offsets.end(),
                           d_ray_offsets.begin());
    size_t allocate_size = d_ray_offsets[n_rays-1] + last_ray_hitcount;
    // Each ray segment in the output arrays ends with a dummy, or sentinel,
    // value marking the end of the ray; increase offsets accordingly.
    // transform:
    //     => offsets = [0, 4, 5, 10]
    allocate_size += n_rays;
    thrust::transform(d_ray_offsets.begin(), d_ray_offsets.end(),
                      thrust::make_counting_iterator(0),
                      d_ray_offsets.begin(),
                      thrust::plus<int>());

    // Initially, outputs should be populated with their sentinel/dummy values,
    // since these are not touched during tracing.
    d_hit_indices.resize(allocate_size, index_sentinel);
    d_hit_integrals.resize(allocate_size, integral_sentinel);
    d_hit_distances.resize(allocate_size, distance_sentinel);

    // TODO: Change it such that this is passed in, rather than copying it on
    // each call.
    const size_t sm_table_size = sizeof(double) * N_table;
    const double* p_table = &(lookup.table[0]);
    thrust::device_vector<double> d_lookup(p_table, p_table + N_table);

    typedef RayData_sphere<int, Real> RayData;
    trace_texref<RayData>(
        d_rays,
        d_spheres,
        d_tree,
        sm_table_size,
        InitGlobalToSmem<double>(
            thrust::raw_pointer_cast(d_lookup.data()),
            N_table),
        Intersect_sphere_b2dist(),
        OnHit_sphere_individual<IndexType, Real>(
            thrust::raw_pointer_cast(d_hit_indices.data()),
            thrust::raw_pointer_cast(d_hit_integrals.data()),
            thrust::raw_pointer_cast(d_hit_distances.data()),
            N_table),
        RayEntry_from_array<int>(
            thrust::raw_pointer_cast(d_ray_offsets.data())),
        RayExit_null()
    );
}

} // namespace grace
