#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"
#include "../ray.h"


namespace grace {

//-----------------------------------------------------------------------------
// Helper functions for tracing kernels
//----------------------------------------------------------------------------
#define N_TABLE 51
float kernel_integral_table[N_TABLE] = {
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
    0.000000000000000E+000};

enum CLASSIFICATION
{ MMM, PMM, MPM, PPM, MMP, PMP, MPP, PPP };

__host__ __device__ SlopeProp slope_properties(const Ray& ray)
{
    SlopeProp slope;

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
                                           const SlopeProp& slope,
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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false; // past length of ray

        else if ((ybyx * tx - by + c_xy < 0.0f) ||
                 (xbyy * ty - bx + c_yx < 0.0f) ||
                 // ( (bx > ox) && (ybyx * bx - ty + c_xy > 0.0f) ) ||
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
                                    const Float4& xyzr,
                                    Float& b,
                                    Float& dot_p)
{
    // Ray origin -> sphere centre.
    float px = xyzr.x - ray.ox;
    float py = xyzr.y - ray.oy;
    float pz = xyzr.z - ray.oz;

    // Normalized ray direction.
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;

    // Projection of p onto r, i.e. distance to intersection.
    dot_p = px*rx + py*ry + pz*rz;

    // Impact parameter.
    float bx = px - dot_p*rx;
    float by = py - dot_p*ry;
    float bz = pz - dot_p*rz;
    b = sqrtf(bx*bx + by*by +bz*bz);

    if (b >= xyzr.w)
        return false;

    // If dot_p < 0, the ray origin must be inside the sphere for an
    // intersection. We treat this edge-case as a miss.
    if (dot_p < 0)
        return false;

    // If dot_p > ray length, the ray terminus must be inside the sphere for
    // an intersection. We treat this edge-case as a miss.
    if (dot_p > ray.length)
        return false;

    // Otherwise, assume we have a hit.  This counts the following partial
    // intersections as hits:
    //     i) Ray starts inside sphere, before point of closest approach.
    //    ii) Ray ends inside sphere, beyond point of closest approach.
    return true;
}

//-----------------------------------------------------------------------------
// CUDA tracing kernels
//-----------------------------------------------------------------------------

namespace gpu {

// Trace through the field, but save only the number of hits for each ray.
template <typename Float4>
__global__ void trace_hitcount(const Ray* rays,
                               const size_t n_rays,
                               unsigned int* hit_counts,
                               const int4* nodes,
                               const Box* nodes_AABB,
                               size_t n_nodes,
                               const Float4* xyzrs)
{
    int node_index, ray_index, stack_index;
    unsigned int ray_hit_count;
    int4 node;
    bool is_leaf;
    // Unused in this kernel.
    float b, d;
    // N (31) levels inc. leaves => N-1 (30) key length.
    // One extra so the bottom of the stack can contain a marker that tells
    // us we have emptied the stack.
    // Here that marker is 0 since it must be a valid index of the nodes array!
    //__shared__ int trace_stack[31*TRACE_THREADS_PER_BLOCK];
    int trace_stack[31];
    trace_stack[0] = 0;

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    while (ray_index < n_rays)
    {
        Ray ray = rays[ray_index];
        SlopeProp slope = slope_properties(ray);
        ray_hit_count = 0;

        // Top of the stack.
        // Must provide a valid node index, so points to root.
        //stack_index = threadIdx.x*31;
        stack_index = 0;
        node_index = 0;
        is_leaf = false;

        //while (stack_index >= (int) threadIdx.x*31)
        while (stack_index >= 0)
        {
            while (!is_leaf && stack_index >= 0)
            {
                // If current node is hit, put its right child at the top of
                // the stack and move to its left child.
                if (AABB_hit_eisemann(ray, slope,
                                      nodes_AABB[node_index]))
                {
                    node = nodes[node_index];
                    stack_index++;
                    trace_stack[stack_index] = node.y;

                    node_index = node.x;

                }
                // If it is not hit, move to the node at the top of the stack.
                else
                {
                    node_index = trace_stack[stack_index];
                    stack_index--;
                }

                // Check if the current node is a leaf.
                is_leaf = node_index > n_nodes-1;
            }

            if (is_leaf && stack_index >= 0)
            {
                if (sphere_hit(ray, xyzrs[node_index-n_nodes], b, d))
                {
                    ray_hit_count++;
                }
                // Move to the right child of the node at the top of the stack.
                node_index = trace_stack[stack_index];
                stack_index--;

                is_leaf = node_index > n_nodes-1;
            }

        }
        hit_counts[ray_index] = ray_hit_count;
        ray_index += blockDim.x * gridDim.x;
    }
}

// Trace through the field and accummulate some property for each intersection.
template <typename Tout, typename Float4, typename Tin, typename Float>
__global__ void trace_property(const Ray* rays,
                               const size_t n_rays,
                               Tout* out_data,
                               const int4* nodes,
                               const Box* nodes_AABB,
                               const size_t n_nodes,
                               const Float4* xyzrs,
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
        SlopeProp slope = slope_properties(ray);

        stack_index = 0;
        node_index = 0;
        is_leaf = false;
        out = 0;

        while (stack_index >= 0)
        {
            while (!is_leaf && stack_index >= 0)
            {
                if (AABB_hit_eisemann(ray, slope,
                                      nodes_AABB[node_index]))
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

                is_leaf = node_index > n_nodes-1;
            }

            if (is_leaf && stack_index >= 0)
            {
                if (sphere_hit(ray, xyzrs[node_index-n_nodes], b, d))
                {
                    r = xyzrs[node_index-n_nodes].w;
                    ir = 1.f / r;
                    // Normalize impact parameter, then scale by size of lookup
                    // table and interpolate.
                    b = (N_TABLE-1) * (b * ir);
                    // Floor(b).
                    b_index = (int) b;
                    if (b_index > (N_TABLE-1)) {
                        b = 1.f;
                        b_index = N_TABLE-2;
                    }
                    Float kernel_fac = (  b_integrals[b_index+1]
                                        - b_integrals[b_index]
                                       ) * (b - b_index)
                                       + b_integrals[b_index];
                    // Re-scale integral (since we used a normalized b).
                    kernel_fac *= (ir*ir);
                    out += (Tout) (kernel_fac * p_data[node_index-n_nodes]);
                }
                node_index = trace_stack[stack_index];
                stack_index--;

                is_leaf = node_index > n_nodes-1;
            }

        }
        out_data[ray_index] = out;
        ray_index += blockDim.x * gridDim.x;
    }
}

// Trace through the field and save information for each intersection.
template <typename Tout, typename Float, typename Float4, typename Tin>
__global__ void trace(const Ray* rays,
                      const size_t n_rays,
                      Tout* out_data,
                      Float* hit_dists,
                      const uinteger32* hit_offsets,
                      const int4* nodes,
                      const Box* nodes_AABB,
                      const size_t n_nodes,
                      const Float4* xyzrs,
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
        out_index = hit_offsets[ray_index];
        Ray ray = rays[ray_index];
        SlopeProp slope = slope_properties(ray);

        stack_index = 0;
        node_index = 0;
        is_leaf = false;

        while (stack_index >= 0)
        {
            while (!is_leaf && stack_index >= 0)
            {
                if (AABB_hit_eisemann(ray, slope,
                                      nodes_AABB[node_index]))
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

                is_leaf = node_index > n_nodes-1;
            }

            if (is_leaf && stack_index >= 0)
            {
                if (sphere_hit(ray, xyzrs[node_index-n_nodes], b, d))
                {
                    r = xyzrs[node_index-n_nodes].w;
                    ir = 1.f / r;
                    b = (N_TABLE-1) * (b * ir);
                    b_index = (int) b;
                    if (b_index > (N_TABLE-1)) {
                        b = 1.f;
                        b_index = N_TABLE-2;
                    }
                    Float kernel_fac = (  b_integrals[b_index+1]
                                        - b_integrals[b_index]
                                       ) * (b - b_index)
                                       + b_integrals[b_index];
                    kernel_fac *= (ir*ir);
                    out_data[out_index] = (Tout) (kernel_fac *
                                                  p_data[node_index-n_nodes]);
                    hit_dists[out_index] = d;
                    out_index++;
                }
                node_index = trace_stack[stack_index];
                stack_index--;

                is_leaf = node_index > n_nodes-1;
            }

        }
        ray_index += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tracing kernels
//-----------------------------------------------------------------------------

} // namespace grace
