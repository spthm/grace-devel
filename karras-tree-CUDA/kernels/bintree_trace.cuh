#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"
#include "../ray.h"

namespace grace {

// Using video instructions
__device__ __inline__ int   min_min   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin (float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax (float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin (float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax (float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }


__device__ __inline__ float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
            float t1 = fmin_fmax(a0, a1, d);
            float t2 = fmin_fmax(b0, b1, t1);
            float t3 = fmin_fmax(c0, c1, t2);
            return t3;
}

__device__ __inline__ float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
            float t1 = fmax_fmin(a0, a1, d);
            float t2 = fmax_fmin(b0, b1, t1);
            float t3 = fmax_fmin(c0, c1, t2);
            return t3;
}

__device__ __inline__ float spanBeginFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d) {       return magic_max7(a0, a1, b0, b1, c0, c1, d); }
__device__ __inline__ float spanEndFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d)   {       return magic_min7(a0, a1, b0, b1, c0, c1, d); }

__device__ bool AABB_hit(const Ray& ray, const Node& node)
{
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    float irx = 1. / ray.dx;
    float iry = 1. / ray.dy;
    float irz = 1. / ray.dz;

    // Assume origin at (0, 0, 0)
    float bx = node.bottom[0];
    float by = node.bottom[1];
    float bz = node.bottom[2];
    float tx = node.top[0];
    float ty = node.top[1];
    float tz = node.top[2];

    tmin = (bx) * irx;
    tmax = (tx) * irx;
    tymin = (by) * iry;
    tymax = (ty) * iry;
    tzmin = (bz) * irz;
    tzmax = (tz) * irz;

    tmin = spanBeginFermi(tmin, tmax, tymin, tymax, tzmin, tzmax, 0.0);
    tmax = spanEndFermi(tmin, tmax, tymin, tmax, tzmin, tzmax, ray.length);

    return (tmin <= tmax);
}

__host__ __device__ bool AABB_hit_plucker(const Ray& ray, const Node& node) {
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;
    float length = ray.length;
    float s2bx, s2by, s2bz; // Vector from ray start to lower cell corner.
    float s2tx, s2ty, s2tz; // Vector from ray start to upper cell corner.
    float e2bx, e2by, e2bz; // Vector from ray end to lower cell corner.
    float e2tx, e2ty, e2tz; // Vector from ray end to upper cell corner.

    s2bx = node.bottom[0] - ray.ox;
    s2by = node.bottom[1] - ray.oy;
    s2bz = node.bottom[2] - ray.oz;

    s2tx = node.top[0] - ray.ox;
    s2ty = node.top[1] - ray.oy;
    s2tz = node.top[2] - ray.oz;

    e2bx = s2bx - rx*length;
    e2by = s2by - ry*length;
    e2bz = s2bz - rz*length;

    e2tx = s2tx - rx*length;
    e2ty = s2ty - ry*length;
    e2tz = s2tz - rz*length;

    switch(ray.dclass)
    {
        // MMM
        case 0:
        if (s2bx > 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2ty < 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (rx*s2by - ry*s2tx < 0.0f ||
            rx*s2ty - ry*s2bx > 0.0f ||
            rx*s2tz - rz*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx < 0.0f ||
            ry*s2bz - rz*s2ty < 0.0f ||
            ry*s2tz - rz*s2by > 0.0f ) return false;
        break;

        // PMM
        case 1:
        if (s2tx < 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2ty < 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (rx*s2ty - ry*s2tx < 0.0f ||
            rx*s2by - ry*s2bx > 0.0f ||
            rx*s2bz - rz*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx < 0.0f ||
            ry*s2bz - rz*s2ty < 0.0f ||
            ry*s2tz - rz*s2by > 0.0f) return false;
        break;

        // MPM
        case 2:
        if (s2bx > 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2by > 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (rx*s2by - ry*s2bx < 0.0f ||
            rx*s2ty - ry*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx < 0.0f ||
            ry*s2tz - rz*s2ty < 0.0f ||
            ry*s2bz - rz*s2by > 0.0f) return false;
        break;

        // PPM
        case 3:
        if (s2tx < 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2by > 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (rx*s2ty - ry*s2bx < 0.0f ||
            rx*s2by - ry*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx < 0.0f ||
            ry*s2tz - rz*s2ty < 0.0f ||
            ry*s2bz - rz*s2by > 0.0f) return false;
        break;

        // MMP
        case 4:
        if (s2bx > 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2ty < 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (rx*s2by - ry*s2tx < 0.0f ||
            rx*s2ty - ry*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx < 0.0f ||
            ry*s2bz - rz*s2by < 0.0f ||
            ry*s2tz - rz*s2ty > 0.0f) return false;
        break;

        // PMP
        case 5:
        if (s2tx < 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2ty < 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (rx*s2ty - ry*s2tx < 0.0f ||
            rx*s2by - ry*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx < 0.0f ||
            ry*s2bz - rz*s2by < 0.0f ||
            ry*s2tz - rz*s2ty > 0.0f) return false;
        break;

        // MPP
        case 6:
        if (s2bx > 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2by > 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (rx*s2by - ry*s2bx < 0.0f ||
            rx*s2ty - ry*s2tx > 0.0f ||
            rx*s2tz - rz*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx < 0.0f ||
            ry*s2tz - rz*s2by < 0.0f ||
            ry*s2bz - rz*s2ty > 0.0f) return false;
        break;

        // PPP
        case 7:
        if (s2tx < 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2by > 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (rx*s2ty - ry*s2bx < 0.0f ||
            rx*s2by - ry*s2tx > 0.0f ||
            rx*s2bz - rz*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx < 0.0f ||
            ry*s2tz - rz*s2by < 0.0f ||
            ry*s2bz - rz*s2ty > 0.0f) return false;
        break;
    }
    // Didn't return false above, so we have a hit.
    return true;

}

template <typename Float>
__host__ __device__ bool sphere_hit(const Ray& ray,
                                    const Float& x,
                                    const Float& y,
                                    const Float& z,
                                    const Float& radius)
{
    // Ray origin -> sphere centre;
    float px = ray.ox - x;
    float py = ray.oy - y;
    float pz = ray.oz - z;

    // Normalized ray direction.
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;

    // Projection of p onto r.
    float dot_p = px*rx + py*ry + pz*rz;

    // Impact parameter.
    float bx = px - dot_p*rx;
    float by = py - dot_p*ry;
    float bz = pz - dot_p*rz;
    float b = sqrtf(bx*bx + by*by +bz*bz);

    if (b >= radius)
        return false;

    // If dot_p < 0, to hit the ray origin must be inside the sphere.
    // This is not possible if the distance along the ray (backwards from its
    // origin) to the point of closest approach is > the sphere radius.
    if (dot_p < -radius)
        return false;

    // The ray terminates before piercing the sphere.
    if (dot_p > ray.length + radius)
        return false;

    // Otherwise, assume we have a hit.  This counts the following partial
    // intersections as hits:
    //     i) Ray starts (anywhere) inside sphere.
    //    ii) Ray ends (anywhere) inside sphere.
    return true;
}

namespace gpu {

template <typename Float>
__global__ void trace(const Ray* rays,
                      const int n_rays,
                      const int max_ray_hits,
                      int* hits,
                      int* hit_counts,
                      const Node* nodes,
                      const Leaf* leaves,
                      const Float* xs,
                      const Float* ys,
                      const Float* zs,
                      const Float* radii)
{
    int ray_index, stack_index, hit_offset, ray_hit_count;
    Integer32 node_index;
    bool is_leaf;
    Ray ray;
    // N levels => N-1 key length.
    // One extra so we can avoid stack_index = -1 before trace exit.
    int trace_stack[31];

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;
    // Top of the stack.  Must provide a valid node index, so points to root.
    trace_stack[0] = 0;

    while (ray_index <= n_rays)
    {
        node_index = 0;
        stack_index = 0;
        is_leaf = false;

        hit_offset = ray_index*max_ray_hits;
        ray_hit_count = 0;
        ray = rays[ray_index];

        while (stack_index >= 0)
        {
            if (!is_leaf)
            {
                if (AABB_hit(ray, nodes[node_index])) {
                    stack_index++;
                    trace_stack[stack_index] = node_index;
                    is_leaf = nodes[node_index].left_leaf_flag;
                    node_index = nodes[node_index].left;

                }
                else {
                    node_index = trace_stack[stack_index];
                    stack_index--;
                    is_leaf = nodes[node_index].right_leaf_flag;
                    node_index = nodes[node_index].right;
                }
            }

            if (is_leaf)
            {
                if (sphere_hit(ray,
                               xs[node_index], ys[node_index], zs[node_index],
                               radii[node_index]))
                {
                    if (ray_hit_count < max_ray_hits)
                        hits[hit_offset+ray_hit_count] = node_index;
                    ray_hit_count++;
                }
                node_index = trace_stack[stack_index];
                stack_index--;
                is_leaf = nodes[node_index].right_leaf_flag;
                node_index = nodes[node_index].right;
            }
        }
        hit_counts[ray_index] = ray_hit_count;
        ray_index += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

} // namespace grace
