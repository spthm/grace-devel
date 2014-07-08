#pragma once

#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "../ray.h"


namespace grace {

//-----------------------------------------------------------------------------
// Helper functions for tracing kernels.
//-----------------------------------------------------------------------------

const int N_table = 51;

template <typename Float>
struct KernelIntegrals
{
    static Float table[N_table];

};

template <typename Float>
    Float KernelIntegrals<Float>::table[N_table] = {
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

// min(min(a, b), c)
__device__ __inline__ int min_vmin (int a, int b, int c) {
    int mvm;
    asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// max(max(a, b), c)
__device__ __inline__ int max_vmax (int a, int b, int c) {
    int mvm;
    asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// max(min(a, b), c)
__device__ __inline__ int max_vmin (int a, int b, int c) {
    int mvm;
    asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// min(max(a, b), c)
__device__ __inline__ int min_vmax (int a, int b, int c) {
    int mvm;
    asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}

__device__ __inline__ float minf_vminf (float f1, float f2, float f3) {
    return __int_as_float(min_vmin(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __inline__ float maxf_vmaxf (float f1, float f2, float f3) {
    return __int_as_float(max_vmax(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __inline__ float minf_vmaxf (float f1, float f2, float f3) {
    return __int_as_float(min_vmax(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __inline__ float maxf_vminf (float f1, float f2, float f3) {
    return __int_as_float(max_vmin(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}

__device__ int AABBs_hit(const float3 invd, const float3 ood, const float len,
                         const float4 AABBx,
                         const float4 AABBy,
                         const float4 AABBz)
{
    float bx, tx, by, ty, bz, tz;
    float tmin, tmax;
    unsigned int hits = 0;

    // FMA.
    bx = AABBx.x * invd.x - ood.x;
    tx = AABBx.y * invd.x - ood.x;
    by = AABBy.x * invd.y - ood.y;
    ty = AABBy.y * invd.y - ood.y;
    bz = AABBz.x * invd.z - ood.z;
    tz = AABBz.y * invd.z - ood.z;
    tmin = maxf_vmaxf( fmin(bx, tx), fmin(by, ty), maxf_vminf(bz, tz, 0) );
    tmax = minf_vminf( fmax(bx, tx), fmax(by, ty), minf_vmaxf(bz, tz, len) );
    hits += (int)(tmax >= tmin);

    bx = AABBx.z * invd.x - ood.x;
    tx = AABBx.w * invd.x - ood.x;
    by = AABBy.z * invd.y - ood.y;
    ty = AABBy.w * invd.y - ood.y;
    bz = AABBz.z * invd.z - ood.z;
    tz = AABBz.w * invd.z - ood.z;
    tmin = maxf_vmaxf( fmin(bx, tx), fmin(by, ty), maxf_vminf(bz, tz, 0) );
    tmax = minf_vminf( fmax(bx, tx), fmax(by, ty), minf_vmaxf(bz, tz, len) );
    hits += 2*((int)(tmax >= tmin));

    return hits;
}

template <typename Float4, typename Float>
__host__ __device__ bool sphere_hit(const Ray& ray,
                                    const Float4& sphere,
                                    Float& b,
                                    Float& dot_p)
{
    float px = sphere.x - ray.ox;
    float py = sphere.y - ray.oy;
    float pz = sphere.z - ray.oz;

    // Already normalized.
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;

    // Distance to intersection.
    dot_p = px*rx + py*ry + pz*rz;

    // Impact parameter.
    float bx = px - dot_p*rx;
    float by = py - dot_p*ry;
    float bz = pz - dot_p*rz;
    b = sqrtf(bx*bx + by*by + bz*bz);

    if (b >= sphere.w)
        return false;

    // If dot_p < 0, the ray origin must be inside the sphere for an
    // intersection. We treat this edge-case as a miss.
    if (dot_p < 0.0f)
        return false;

    // If dot_p > ray length, the ray terminus must be inside the sphere for
    // an intersection. We treat this edge-case as a miss.
    if (dot_p >= ray.length)
        return false;

    // Otherwise, assume we have a hit.  This counts the following partial
    // intersections as hits:
    //     i) Ray starts inside sphere, before point of closest approach.
    //    ii) Ray ends inside sphere, beyond point of closest approach.
    return true;
}

namespace gpu {

//-----------------------------------------------------------------------------
// CUDA tracing kernels.
//-----------------------------------------------------------------------------

template <typename Float4>
__global__ void trace_hitcounts_kernel(const Ray* rays,
                                       const size_t n_rays,
                                       unsigned int* hit_counts,
                                       const float4* nodes,
                                       size_t n_nodes,
                                       const int4* leaves,
                                       const Float4* spheres,
                                       const size_t n_spheres)
{
    int node_index, ray_index, stack_index, lr_hit;
    unsigned int ray_hit_count;
    int4 node;
    Ray ray;
    float3 invd, ood;
    float4 AABBx, AABBy, AABBz;
    // Unused in this kernel.
    float b, d;
    // Including leaves there are 31 levels for 30-bit keys.
    int trace_stack[31];
    trace_stack[0] = 0;

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    while (ray_index < n_rays)
    {
        ray = rays[ray_index];
        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        ood.x = ray.ox * invd.x;
        ood.y = ray.oy * invd.y;
        ood.z = ray.oz * invd.z;

        ray_hit_count = 0;

        // Always start at the bottom of the stack (the root node).
        stack_index = 0;
        node_index = 0;

        while (stack_index >= 0)
        {
            while (node_index < n_nodes && stack_index >= 0)
            {
                assert(4*node_index + 3 < 4*n_nodes);
                node = *reinterpret_cast<const int4*>(&nodes[4*node_index + 0]);
                AABBx = nodes[4*node_index + 1];
                AABBy = nodes[4*node_index + 2];
                AABBz = nodes[4*node_index + 3];
                lr_hit = AABBs_hit(invd, ood, ray.length, AABBx, AABBy, AABBz);
                // Most likely to hit both.
                if (lr_hit == 3)
                {
                    // Traverse to left child and push right to stack.
                    node_index = node.x;
                    stack_index++;
                    trace_stack[stack_index] = node.y;
                }
                else
                {
                    // Likely to hit at least one.
                    if (lr_hit)
                    {
                        // Traverse to the only node that was hit.
                        if (lr_hit == 2)
                            node_index = node.y;
                        else
                            node_index = node.x;
                    }
                    else
                    {
                        // Neither hit.  Pop stack.
                        node_index = trace_stack[stack_index];
                        stack_index--;
                    }
                }
            }

            while (node_index >= n_nodes && stack_index >= 0)
            {
                node = leaves[node_index-n_nodes];
                for (int i=0; i<node.y; i++)
                {
                    assert(node.y > 0);
                    assert(node.x+node.y-1 < n_spheres);
                    if (sphere_hit(ray, spheres[node.x+i], b, d))
                    {
                        ray_hit_count++;
                    }
                }
                // Pop stack.
                node_index = trace_stack[stack_index];
                stack_index--;
            }

        }
        hit_counts[ray_index] = ray_hit_count;
        ray_index += blockDim.x * gridDim.x;
    }
}

template <typename Tout, typename Float4, typename Tin, typename Float>
__global__ void trace_property_kernel(const Ray* rays,
                                      const size_t n_rays,
                                      Tout* out_data,
                                      const float4* nodes,
                                      const size_t n_nodes,
                                      const int4* leaves,
                                      const Float4* spheres,
                                      const Tin* p_data,
                                      const Float* b_integrals)
{
    int node_index, ray_index, stack_index, lr_hit, b_index;
    int4 node;
    Ray ray;
    float3 invd, ood;
    float4 AABBx, AABBy, AABBz;
    // Impact parameter and distance to intersection.
    float b, d;
    // Sphere radius and 1/radius.
    float r, ir;
    // Property to trace and accumulate.
    Tout out;
    int trace_stack[31];
    trace_stack[0] = 0;

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    while (ray_index < n_rays)
    {
        ray = rays[ray_index];
        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        ood.x = ray.ox * invd.x;
        ood.y = ray.oy * invd.y;
        ood.z = ray.oz * invd.z;

        stack_index = 0;
        node_index = 0;
        out = 0;

        while (stack_index >= 0)
        {
            while (node_index < n_nodes && stack_index >= 0)
            {
                node = *reinterpret_cast<const int4*>(&nodes[4*node_index + 0]);
                AABBx = nodes[4*node_index + 1];
                AABBy = nodes[4*node_index + 2];
                AABBz = nodes[4*node_index + 3];
                lr_hit = AABBs_hit(invd, ood, ray.length, AABBx, AABBy, AABBz);
                if (lr_hit == 3)
                {
                    node_index = node.x;
                    stack_index++;
                    trace_stack[stack_index] = node.y;
                }
                else
                {
                    if (lr_hit)
                    {
                        if (lr_hit == 2)
                            node_index = node.y;
                        else
                            node_index = node.x;
                    }
                    else
                    {
                        node_index = trace_stack[stack_index];
                        stack_index--;
                    }
                }
            }

            while (node_index >= n_nodes && stack_index >= 0)
            {
                node = leaves[node_index-n_nodes];
                for (int i=0; i<node.y; i++)
                {
                    if (sphere_hit(ray, spheres[node.x+i], b, d))
                    {
                        r = spheres[node.x+i].w;
                        ir = 1.f / r;
                        // Normalize impact parameter to size of lookup table and
                        // interpolate.
                        b = (N_table-1) * (b * ir);
                        b_index = (int) b; // == floor(b)
                        if (b_index > (N_table-1)) {
                            b = 1.f;
                            b_index = N_table-2;
                        }
                        Float kernel_fac =
                            (b_integrals[b_index+1] - b_integrals[b_index])
                            * (b - b_index)
                            + b_integrals[b_index];
                        // Re-scale integral (since we used a normalized b).
                        kernel_fac *= (ir*ir);
                        out += (Tout) (kernel_fac * p_data[node.x+i]);
                    }
                }
                node_index = trace_stack[stack_index];
                stack_index--;
            }
        }
        out_data[ray_index] = out;
        ray_index += blockDim.x * gridDim.x;
    }
}

template <typename Tout, typename Float, typename Float4, typename Tin>
__global__ void trace_kernel(const Ray* rays,
                             const size_t n_rays,
                             Tout* out_data,
                             unsigned int* hit_indices,
                             Float* hit_distances,
                             const unsigned int* ray_offsets,
                             const float4* nodes,
                             const size_t n_nodes,
                             const int4* leaves,
                             const Float4* spheres,
                             const Tin* p_data,
                             const Float* b_integrals)
{
    int node_index, ray_index, stack_index, lr_hit, b_index;
    unsigned int out_index;
    int4 node;
    Ray ray;
    float3 invd, ood;
    float4 AABBx, AABBy, AABBz;
    float b, d;
    float r, ir;
    int trace_stack[31];
    trace_stack[0] = 0;

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    while (ray_index < n_rays)
    {
        out_index = ray_offsets[ray_index];
        ray = rays[ray_index];
        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        ood.x = ray.ox * invd.x;
        ood.y = ray.oy * invd.y;
        ood.z = ray.oz * invd.z;

        stack_index = 0;
        node_index = 0;

        while (stack_index >= 0)
        {
            while (node_index < n_nodes && stack_index >= 0)
            {
                node = *reinterpret_cast<const int4*>(&nodes[4*node_index + 0]);
                AABBx = nodes[4*node_index + 1];
                AABBy = nodes[4*node_index + 2];
                AABBz = nodes[4*node_index + 3];
                lr_hit = AABBs_hit(invd, ood, ray.length, AABBx, AABBy, AABBz);
                if (lr_hit == 3)
                {
                    node_index = node.x;
                    stack_index++;
                    trace_stack[stack_index] = node.y;
                }
                else
                {
                    if (lr_hit)
                    {
                        if (lr_hit == 2)
                            node_index = node.y;
                        else
                            node_index = node.x;
                    }
                    else
                    {
                        node_index = trace_stack[stack_index];
                        stack_index--;
                    }
                }
            }

            while (node_index >= n_nodes && stack_index >= 0)
            {
                node = leaves[node_index-n_nodes];
                for (int i=0; i<node.y; i++)
                {
                    if (sphere_hit(ray, spheres[node.x+i], b, d))
                    {
                        r = spheres[node.x+i].w;
                        ir = 1.f / r;
                        b = (N_table-1) * (b * ir);
                        b_index = (int) b;
                        if (b_index > (N_table-1)) {
                            b = 1.f;
                            b_index = N_table-2;
                        }
                        Float kernel_fac =
                            (b_integrals[b_index+1] - b_integrals[b_index])
                            * (b - b_index)
                            + b_integrals[b_index];
                        kernel_fac *= (ir*ir);
                        out_data[out_index] =
                            (Tout) (kernel_fac * p_data[node.x+i]);
                        hit_indices[out_index] = node.x+i;
                        hit_distances[out_index] = d;
                        out_index++;
                    }
                }
                node_index = trace_stack[stack_index];
                stack_index--;
            }
        }
        ray_index += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tracing kernels
//-----------------------------------------------------------------------------

template <typename Float4>
void trace_hitcounts(const thrust::device_vector<Ray>& d_rays,
                     thrust::device_vector<unsigned int>& d_hit_counts,
                     const Tree& d_tree,
                     const thrust::device_vector<Float4>& d_spheres)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    int blocks = min(MAX_BLOCKS, (int) ((n_rays + TRACE_THREADS_PER_BLOCK-1)
                                        / TRACE_THREADS_PER_BLOCK));

    gpu::trace_hitcounts_kernel<<<blocks, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_hit_counts.data()),
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        thrust::raw_pointer_cast(d_spheres.data()),
        d_spheres.size());
}

template <typename Float, typename Tout, typename Float4, typename Tin>
void trace_property(const thrust::device_vector<Ray>& d_rays,
                    thrust::device_vector<Tout>& d_out_data,
                    const Tree& d_tree,
                    const thrust::device_vector<Float4>& d_spheres,
                    const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    // TODO: Change it such that this is passed in, rather than instantiating
    // and copying it on each call to trace_property and trace.
    // Or make it static and initalize it in e.g. a grace_init function, that
    // could also determine kernel launch parameters.
    const KernelIntegrals<Float> lookup;
    const Float* p_table = &(lookup.table[0]);
    thrust::device_vector<Float> d_lookup(p_table, p_table + N_table);

    int blocks = min(MAX_BLOCKS, (int) ((n_rays + TRACE_THREADS_PER_BLOCK-1)
                                        / TRACE_THREADS_PER_BLOCK));

    gpu::trace_property_kernel<<<blocks, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_out_data.data()),
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_lookup.data()));
}

// TODO: Allow the user to supply correctly-sized hit-distance and output
//       arrays to this function, handling any memory issues therein themselves.
template <typename Float, typename Tout, typename Float4, typename Tin>
void trace(const thrust::device_vector<Ray>& d_rays,
           thrust::device_vector<Tout>& d_out_data,
           thrust::device_vector<unsigned int>& d_ray_offsets,
           thrust::device_vector<unsigned int>& d_hit_indices,
           thrust::device_vector<Float>& d_hit_distances,
           const Tree& d_tree,
           const thrust::device_vector<Float4>& d_spheres,
           const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    // Here, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts(d_rays, d_ray_offsets, d_tree, d_spheres);
    unsigned int last_ray_hits = d_ray_offsets[n_rays-1];

    // Allocate output array based on per-ray hit counts, and calculate
    // individual ray offsets into this array:
    //
    // hits = [3, 0, 4, 1]
    // exclusive_scan:
    //    => offsets = [0, 3, 3, 7]
    // total_hits = hits[3] + offsets[3] = 7 + 1 = 8
    thrust::exclusive_scan(d_ray_offsets.begin(), d_ray_offsets.end(),
                           d_ray_offsets.begin());
    unsigned int total_hits = d_ray_offsets[n_rays-1] + last_ray_hits;

    d_out_data.resize(total_hits);
    d_hit_indices.resize(total_hits);
    d_hit_distances.resize(total_hits);

    // TODO: Change it such that this is passed in, rather than instantiating
    // and copying it on each call to trace_property and trace.
    // Or initialize it as static, above, for both float and double.
    // Also then copy it into device memory?
    const KernelIntegrals<Float> lookup;
    const Float* p_table = &(lookup.table[0]);
    thrust::device_vector<Float> d_lookup(p_table, p_table + N_table);

    int blocks = min(MAX_BLOCKS, (int) ((n_rays + TRACE_THREADS_PER_BLOCK-1)
                                        / TRACE_THREADS_PER_BLOCK));

    gpu::trace_kernel<<<blocks, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_out_data.data()),
        thrust::raw_pointer_cast(d_hit_indices.data()),
        thrust::raw_pointer_cast(d_hit_distances.data()),
        thrust::raw_pointer_cast(d_ray_offsets.data()),
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_lookup.data()));
}

// TODO: Break this and trace() into multiple functions.
template <typename Float, typename Tout, typename Float4, typename Tin>
void trace_with_sentinels(const thrust::device_vector<Ray>& d_rays,
                          thrust::device_vector<Tout>& d_out_data,
                          const Tout out_sentinel,
                          thrust::device_vector<unsigned int>& d_ray_offsets,
                          thrust::device_vector<unsigned int>& d_hit_indices,
                          const unsigned int hit_sentinel,
                          thrust::device_vector<Float>& d_hit_distances,
                          const Float distance_sentinel,
                          const Tree& d_tree,
                          const thrust::device_vector<Float4>& d_spheres,
                          const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    // Here, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts(d_rays, d_ray_offsets, d_tree, d_spheres);
    unsigned int last_ray_hits = d_ray_offsets[n_rays-1];

    // Allocate output array based on per-ray hit counts, and calculate
    // individual ray offsets into this array:
    //
    // hits = [3, 0, 4, 1]
    // exclusive_scan:
    //     => offsets = [0, 3, 3, 7]
    thrust::exclusive_scan(d_ray_offsets.begin(), d_ray_offsets.end(),
                           d_ray_offsets.begin());
    size_t allocate_size = d_ray_offsets[n_rays-1] + last_ray_hits;
    // Increase the offsets such that they are correct if each ray segment in
    // the output array(s) ends with a dummy particle, or sentinel value,
    // marking the end of that ray's data.
    // transform:
    //     => offsets = [0, 4, 5, 10]
    allocate_size += n_rays; // For ray-end markers.
    {
    thrust::device_vector<unsigned int> d_offsets_inc(n_rays);
    thrust::sequence(d_offsets_inc.begin(), d_offsets_inc.end(), 0u);
    thrust::transform(d_ray_offsets.begin(), d_ray_offsets.end(),
                      d_offsets_inc.begin(),
                      d_ray_offsets.begin(),
                      thrust::plus<unsigned int>());
    } // So temporary d_offsets_inc is destroyed.

    // Initially, outputs should be populated with their sentinel/dummy values,
    // since these are not touched during tracing.
    d_out_data.resize(allocate_size, out_sentinel);
    d_hit_indices.resize(allocate_size, hit_sentinel);
    d_hit_distances.resize(allocate_size, distance_sentinel);

    // TODO: Change it such that this is passed in, rather than instantiating
    // and copying it on each call to trace_property and trace.
    // Or initialize it as static, above, for both float and double.
    // Also then copy it into device memory?
    const KernelIntegrals<Float> lookup;
    const Float* p_table = &(lookup.table[0]);
    thrust::device_vector<Float> d_lookup(p_table, p_table + N_table);

    int blocks = min(MAX_BLOCKS, (int) ((n_rays + TRACE_THREADS_PER_BLOCK-1)
                                        / TRACE_THREADS_PER_BLOCK));

    gpu::trace_kernel<<<blocks, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_out_data.data()),
        thrust::raw_pointer_cast(d_hit_indices.data()),
        thrust::raw_pointer_cast(d_hit_distances.data()),
        thrust::raw_pointer_cast(d_ray_offsets.data()),
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_lookup.data()));
}

} // namespace grace
