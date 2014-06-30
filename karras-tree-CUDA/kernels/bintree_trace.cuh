#pragma once

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

enum DIR_CLASS
{ MMM, PMM, MPM, PPM, MMP, PMP, MPP, PPP };

__host__ __device__ RaySlope ray_slope(const Ray& ray)
{
    RaySlope slope;

    slope.xbyy = ray.dx / ray.dy;
    slope.ybyx = 1.0f / slope.xbyy;
    slope.ybyz = ray.dy / ray.dz;
    slope.zbyy = 1.0f / slope.ybyz;
    slope.xbyz = ray.dx / ray.dz;
    slope.zbyx = 1.0f / slope.xbyz;

    slope.c_xy = ray.oy - slope.ybyx*ray.ox;
    slope.c_xz = ray.oz - slope.zbyx*ray.ox;
    slope.c_yx = ray.ox - slope.xbyy*ray.oy;
    slope.c_yz = ray.oz - slope.zbyy*ray.oy;
    slope.c_zx = ray.ox - slope.xbyz*ray.oz;
    slope.c_zy = ray.oy - slope.ybyz*ray.oz;

    return slope;
}

__host__ __device__ bool AABB_hit_eisemann(const Ray& ray,
                                           const RaySlope& slope,
                                           const Box& node_AABB)
{

    float ox = ray.ox;
    float oy = ray.oy;
    float oz = ray.oz;

    float dx = ray.dx;
    float dy = ray.dy;
    float dz = ray.dz;

    float l = ray.length;

    float tx = node_AABB.tx;
    float ty = node_AABB.ty;
    float tz = node_AABB.tz;
    float bx = node_AABB.bx;
    float by = node_AABB.by;
    float bz = node_AABB.bz;

    float xbyy = slope.xbyy;
    float ybyx = slope.ybyx;
    float ybyz = slope.ybyz;
    float zbyy = slope.zbyy;
    float xbyz = slope.xbyz;
    float zbyx = slope.zbyx;
    float c_xy = slope.c_xy;
    float c_xz = slope.c_xz;
    float c_yx = slope.c_yx;
    float c_yz = slope.c_yz;
    float c_zx = slope.c_zx;
    float c_zy = slope.c_zy;

    switch(ray.dclass)
    {
    case MMM:

        if ((ox < bx) || (oy < by) || (oz < bz))
            return false; // AABB entirely in wrong octant wrt ray origin.

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false; // Past length of ray.

        else if ((ybyx * bx - ty + c_xy > 0.0f) ||
                 (xbyy * by - tx + c_yx > 0.0f) ||
                 (ybyz * bz - ty + c_zy > 0.0f) ||
                 (zbyy * by - tz + c_yz > 0.0f) ||
                 (zbyx * bx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PMM:

        if ((ox > tx) || (oy < by) || (oz < bz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false;
        else if ((ybyx * tx - ty + c_xy > 0.0f) ||
                 (xbyy * by - bx + c_yx < 0.0f) ||
                 (ybyz * bz - ty + c_zy > 0.0f) ||
                 (zbyy * by - tz + c_yz > 0.0f) ||
                 (zbyx * tx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - bx + c_zx < 0.0f))
            return false;

        return true;

    case MPM:

        if ((ox < bx) || (oy > ty) || (oz < bz))
            return false;

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false;
        else if ((ybyx * bx - by + c_xy < 0.0f) ||
                 (xbyy * ty - tx + c_yx > 0.0f) ||
                 (ybyz * bz - by + c_zy < 0.0f) ||
                 (zbyy * ty - tz + c_yz > 0.0f) ||
                 (zbyx * bx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PPM:

        if ((ox > tx) || (oy > ty) || (oz < bz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false;
        else if ((ybyx * tx - by + c_xy < 0.0f) ||
                 (xbyy * ty - bx + c_yx < 0.0f) ||
                 (ybyz * bz - by + c_zy < 0.0f) ||
                 (zbyy * ty - tz + c_yz > 0.0f) ||
                 (zbyx * tx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - bx + c_zx < 0.0f))
            return false;

        return true;

    case MMP:

        if ((ox < bx) || (oy < by) || (oz > tz))
            return false;

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * bx - ty + c_xy > 0.0f) ||
                 (xbyy * by - tx + c_yx > 0.0f) ||
                 (ybyz * tz - ty + c_zy > 0.0f) ||
                 (zbyy * by - bz + c_yz < 0.0f) ||
                 (zbyx * bx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PMP:

        if ((ox > tx) || (oy < by) || (oz > tz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * tx - ty + c_xy > 0.0f) ||
                 (xbyy * by - bx + c_yx < 0.0f) ||
                 (ybyz * tz - ty + c_zy > 0.0f) ||
                 (zbyy * by - bz + c_yz < 0.0f) ||
                 (zbyx * tx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - bx + c_zx < 0.0f))
            return false;

        return true;

    case MPP:

        if ((ox < bx) || (oy > ty) || (oz > tz))
            return false;

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * bx - by + c_xy < 0.0f) ||
                 (xbyy * ty - tx + c_yx > 0.0f) ||
                 (ybyz * tz - by + c_zy < 0.0f) ||
                 (zbyy * ty - bz + c_yz < 0.0f) ||
                 (zbyx * bx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PPP:

        if ((ox > tx) || (oy > ty) || (oz > tz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * tx - by + c_xy < 0.0f) ||
                 (xbyy * ty - bx + c_yx < 0.0f) ||
                 (ybyz * tz - by + c_zy < 0.0f) ||
                 (zbyy * ty - bz + c_yz < 0.0f) ||
                 (zbyx * tx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - bx + c_zx < 0.0f))
            return false;

        return true;
    }

    return false;
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
                                       const int4* nodes,
                                       const Box* node_AABBs,
                                       size_t n_nodes,
                                       const int4* leaves,
                                       const Float4* spheres,
                                       const size_t n_spheres)
{
    int node_index, ray_index, stack_index;
    unsigned int ray_hit_count;
    int4 node;
    bool is_leaf;
    // Unused in this kernel.
    float b, d;
    // Including leaves there are 31 levels for 30-bit keys.
    int trace_stack[31];
    trace_stack[0] = 0;

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    while (ray_index < n_rays)
    {
        Ray ray = rays[ray_index];
        RaySlope slope = ray_slope(ray);
        ray_hit_count = 0;

        // Always start at the bottom of the stack (the root node).
        stack_index = 0;
        node_index = 0;
        is_leaf = false;

        while (stack_index >= 0)
        {
            while (!is_leaf && stack_index >= 0)
            {
                if (AABB_hit_eisemann(ray, slope,
                                      node_AABBs[node_index]))
                {
                    // Hit.  Traverse to left child and push right to stack.
                    node = nodes[node_index];
                    stack_index++;
                    trace_stack[stack_index] = node.y;

                    node_index = node.x;

                }
                else
                {
                    // Missed.  Pop stack.
                    node_index = trace_stack[stack_index];
                    stack_index--;
                }

                is_leaf = node_index >= n_nodes;
            }

            if (is_leaf && stack_index >= 0)
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

                is_leaf = node_index >= n_nodes;
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
                                      const int4* nodes,
                                      const Box* node_AABBs,
                                      const size_t n_nodes,
                                      const int4* leaves,
                                      const Float4* spheres,
                                      const Tin* p_data,
                                      const Float* b_integrals)
{
    int node_index, ray_index, stack_index, b_index;
    int4 node;
    bool is_leaf;
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
        Ray ray = rays[ray_index];
        RaySlope slope = ray_slope(ray);

        stack_index = 0;
        node_index = 0;
        is_leaf = false;
        out = 0;

        while (stack_index >= 0)
        {
            while (!is_leaf && stack_index >= 0)
            {
                if (AABB_hit_eisemann(ray, slope,
                                      node_AABBs[node_index]))
                {
                    node = nodes[node_index];
                    stack_index++;
                    trace_stack[stack_index] = node.y;

                    node_index = node.x;

                }
                else
                {
                    node_index = trace_stack[stack_index];
                    stack_index--;
                }

                is_leaf = node_index >= n_nodes;
            }

            if (is_leaf && stack_index >= 0)
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

                is_leaf = node_index >= n_nodes;
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
                             const int4* nodes,
                             const Box* node_AABBs,
                             const size_t n_nodes,
                             const int4* leaves,
                             const Float4* spheres,
                             const Tin* p_data,
                             const Float* b_integrals)
{
    int node_index, ray_index, stack_index, b_index;
    unsigned int out_index;
    int4 node;
    bool is_leaf;
    float b, d;
    float r, ir;
    int trace_stack[31];
    trace_stack[0] = 0;

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    while (ray_index < n_rays)
    {
        out_index = ray_offsets[ray_index];
        Ray ray = rays[ray_index];
        RaySlope slope = ray_slope(ray);

        stack_index = 0;
        node_index = 0;
        is_leaf = false;

        while (stack_index >= 0)
        {
            while (!is_leaf && stack_index >= 0)
            {
                if (AABB_hit_eisemann(ray, slope,
                                      node_AABBs[node_index]))
                {
                    node = nodes[node_index];
                    stack_index++;
                    trace_stack[stack_index] = node.y;

                    node_index = node.x;

                }
                else
                {
                    node_index = trace_stack[stack_index];
                    stack_index--;
                }

                is_leaf = node_index >= n_nodes;
            }

            if (is_leaf && stack_index >= 0)
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

                is_leaf = node_index >= n_nodes;
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
                     const Nodes& d_nodes,
                     const Leaves& d_leaves,
                     const thrust::device_vector<Float4>& d_spheres)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_nodes.hierarchy.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_rays + TRACE_THREADS_PER_BLOCK-1)
                                        / TRACE_THREADS_PER_BLOCK));

    gpu::trace_hitcounts_kernel<<<blocks, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_hit_counts.data()),
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_leaves.indices.data()),
        thrust::raw_pointer_cast(d_spheres.data()),
        d_spheres.size());
}

template <typename Float, typename Tout, typename Float4, typename Tin>
void trace_property(const thrust::device_vector<Ray>& d_rays,
                    thrust::device_vector<Tout>& d_out_data,
                    const Nodes& d_nodes,
                    const Leaves& d_leaves,
                    const thrust::device_vector<Float4>& d_spheres,
                    const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_nodes.hierarchy.size();

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
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_leaves.indices.data()),
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
           const Nodes& d_nodes,
           const Leaves& d_leaves,
           const thrust::device_vector<Float4>& d_spheres,
           const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_nodes.hierarchy.size();

    // Here, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts(d_rays, d_ray_offsets, d_nodes, d_spheres);
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
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_leaves.indices.data()),
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
                          const Nodes& d_nodes,
                          const Leaves& d_leaves,
                          const thrust::device_vector<Float4>& d_spheres,
                          const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_nodes.hierarchy.size();

    // Here, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts(d_rays, d_ray_offsets, d_nodes, d_spheres);
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
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_leaves.indices.data()),
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_lookup.data()));
}

} // namespace grace
