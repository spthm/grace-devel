#pragma once

#include "grace/cuda/nodes.h"
#include "grace/cuda/detail/functors/trace.cuh"
#include "grace/cuda/detail/kernels/bintree_trace.cuh"
#include "grace/generic/meta.h"
#include "grace/generic/raydata.h"
#include "grace/error.h"
#include "grace/ray.h"
#include "grace/sphere.h"
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

const static int N_table = 81;

template <typename Real>
struct KernelIntegrals
{
    const static Real table[N_table];

};

template <typename Real>
    const Real KernelIntegrals<Real>::table[N_table] = {
    1.90985932e+00,   1.78322068e+00,   1.66556632e+00,   1.55546178e+00,
    1.45212870e+00,   1.35501360e+00,   1.26368166e+00,   1.17777168e+00,
    1.09697252e+00,   1.02100907e+00,   9.49633282e-01,   8.82618007e-01,
    8.19752463e-01,   7.60838898e-01,   7.05689912e-01,   6.54126257e-01,
    6.05974923e-01,   5.61067317e-01,   5.19237405e-01,   4.80319421e-01,
    4.44144244e-01,   4.10529692e-01,   3.79283618e-01,   3.50229047e-01,
    3.23205794e-01,   2.98068319e-01,   2.74683926e-01,   2.52931242e-01,
    2.32698928e-01,   2.13884574e-01,   1.96393751e-01,   1.80139186e-01,
    1.65040056e-01,   1.51021358e-01,   1.38013369e-01,   1.25951163e-01,
    1.14774188e-01,   1.04425889e-01,   9.48533741e-02,   8.60071150e-02,
    7.78406808e-02,   7.03104972e-02,   6.33756315e-02,   5.69975970e-02,
    5.11401777e-02,   4.57692679e-02,   4.08527278e-02,   3.63602516e-02,
    3.22632467e-02,   2.85347244e-02,   2.51491992e-02,   2.20825965e-02,
    1.93121687e-02,   1.68164168e-02,   1.45750197e-02,   1.25687680e-02,
    1.07795036e-02,   9.19006382e-03,   7.78422973e-03,   6.54667836e-03,
    5.46293926e-03,   4.51935346e-03,   3.70303612e-03,   3.00184208e-03,
    2.40433401e-03,   1.89975335e-03,   1.47799351e-03,   1.12957571e-03,
    8.45626941e-04,   6.17860410e-04,   4.38558221e-04,   3.00556558e-04,
    1.97233453e-04,   1.22499459e-04,   7.07916806e-05,   3.70719597e-05,
    1.68303381e-05,   6.09640448e-06,   1.46236018e-06,   1.25527528e-07,
    0.00000000e+00
};

const static KernelIntegrals<double> lookup = {};


//-----------------------------------------------------------------------------
// C-like convenience wrappers for common forms of the tracing kernel.
//-----------------------------------------------------------------------------

// T defines the internal precision of the intersection test.
template <typename T>
GRACE_HOST void trace_hitcounts_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<int>& d_hit_counts)
{
    // Defines only RayData.data, of type int.
    typedef RayData_datum<int> RayData;

    trace_texref<RayData, LeafTraversal::ParallelRays>(
        d_rays,
        d_spheres,
        d_tree,
        0,
        Init_null(),
        Intersect_sphere_bool<T>(),
        OnHit_increment(),
        RayEntry_null(),
        RayExit_to_array<int>(
            thrust::raw_pointer_cast(d_hit_counts.data()))
    );
}

// T defines the internal precision of the intersection test, and of
// intermediate computations for the output(s).
template <typename T, typename OutType>
GRACE_HOST void trace_cumulative_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<OutType>& d_cumulated)
{
    // TODO: Change it such that this is passed in, rather than copying it on
    // each call.
    const size_t sm_table_size = sizeof(double) * N_table;
    const double* p_table = &(lookup.table[0]);
    thrust::device_vector<double> d_lookup(p_table, p_table + N_table);

    typedef RayData_sphere<OutType, OutType> RayData;
    trace_texref<RayData, LeafTraversal::ParallelRays>(
        d_rays,
        d_spheres,
        d_tree,
        sm_table_size,
        InitGlobalToSmem<double>(
            thrust::raw_pointer_cast(d_lookup.data()),
            N_table),
        Intersect_sphere_b2dist<T>(),
        OnHit_sphere_cumulate<T>(N_table),
        RayEntry_null(),
        RayExit_to_array<OutType>(
            thrust::raw_pointer_cast(d_cumulated.data()))
    );
}

// T defines the internal precision of the intersection test, and of
// intermediate computations for the output(s).
template <typename T, typename IndexType, typename OutType>
GRACE_HOST void trace_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require ray offsets to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    thrust::device_vector<OutType>& d_hit_integrals,
    thrust::device_vector<OutType>& d_hit_distances)
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
    //     => offsets = [0, 3, 3, 7]
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

    typedef RayData_sphere<int, OutType> RayData;
    trace_texref<RayData, LeafTraversal::ParallelPrimitives>(
        d_rays,
        d_spheres,
        d_tree,
        sm_table_size,
        InitGlobalToSmem<double>(
            thrust::raw_pointer_cast(d_lookup.data()),
            N_table),
        Intersect_sphere_b2dist<T>(),
        OnHit_sphere_individual<T, IndexType, OutType>(
            thrust::raw_pointer_cast(d_hit_indices.data()),
            thrust::raw_pointer_cast(d_hit_integrals.data()),
            thrust::raw_pointer_cast(d_hit_distances.data()),
            N_table),
        RayEntry_from_array<int>(
            thrust::raw_pointer_cast(d_ray_offsets.data())),
        RayExit_null()
    );
}

// T defines the internal precision of the intersection test, and of
// intermediate computations for the output(s).
template <typename T, typename IndexType, typename OutType>
GRACE_HOST void trace_with_sentinels_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require this to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    const int index_sentinel,
    thrust::device_vector<OutType>& d_hit_integrals,
    const OutType integral_sentinel,
    thrust::device_vector<OutType>& d_hit_distances,
    const OutType distance_sentinel)
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

    typedef RayData_sphere<int, OutType> RayData;
    trace_texref<RayData, LeafTraversal::ParallelPrimitives>(
        d_rays,
        d_spheres,
        d_tree,
        sm_table_size,
        InitGlobalToSmem<double>(
            thrust::raw_pointer_cast(d_lookup.data()),
            N_table),
        Intersect_sphere_b2dist<T>(),
        OnHit_sphere_individual<T, IndexType, OutType>(
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
