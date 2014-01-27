#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"
#include "../ray.h"

namespace grace {

enum CLASSIFICATION
{ MMM, PMM, MPM, PPM, MMP, PMP, MPP, PPP };

__inline__ __host__ __device__ SlopeProp slope_properties(const Ray& ray)
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
                                           const Node& node)
{

    float ox = ray.ox;
    float oy = ray.oy;
    float oz = ray.oz;

    float dx = ray.dx;
    float dy = ray.dy;
    float dz = ray.dz;

    float l = ray.length;

    float bx = node.bottom[0];
    float by = node.bottom[1];
    float bz = node.bottom[2];
    float tx = node.top[0];
    float ty = node.top[1];
    float tz = node.top[2];

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

__host__ __device__ bool AABB_hit_plucker(const Ray& ray, const Node& node)
{
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;
    float length = ray.length;
    float s2bx, s2by, s2bz; // Vector from ray start to lower cell corner.
    float s2tx, s2ty, s2tz; // Vector from ray start to upper cell corner.

    s2bx = node.bottom[0] - ray.ox;
    s2by = node.bottom[1] - ray.oy;
    s2bz = node.bottom[2] - ray.oz;

    s2tx = node.top[0] - ray.ox;
    s2ty = node.top[1] - ray.oy;
    s2tz = node.top[2] - ray.oz;

    switch(ray.dclass)
    {
        // MMM
        case 0:
        if (s2bx > 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2by - ry*length > 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2by - ry*length > 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2by - ry*length > 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

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
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2by - ry*length > 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

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
                                    const Float x,
                                    const Float y,
                                    const Float z,
                                    const Float radius)
{
    // Ray origin -> sphere centre.
    float px = x - ray.ox;
    float py = y - ray.oy;
    float pz = z - ray.oz;

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
    // N (31) levels inc. leaves => N-1 (30) key length.
    // One extra so the bottom of the stack can contain a marker that tells
    // us we have emptied the stack.
    // Here that marker is 0 since it must be a valid index of the nodes array!
    //__shared__ int trace_stack[31*TRACE_THREADS_PER_BLOCK];
    int trace_stack[31];

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    while (ray_index < n_rays)
    {
        // Top of the stack.
        // Must provide a valid node index, so points to root.
        //stack_index = threadIdx.x*31;
        stack_index = 0;

        node_index = 0;
        is_leaf = false;

        hit_offset = ray_index*max_ray_hits;
        ray_hit_count = 0;

        Ray ray = rays[ray_index];
        SlopeProp slope = slope_properties(ray);

        //while (stack_index >= (int) threadIdx.x*31)
        while (stack_index >= 0)
        {
            while (!is_leaf && stack_index >= 0)
            {
                // Test current node for intersection.
                // If it is hit, put it at the top of the stack and move to its
                // left child.
                if (AABB_hit_eisemann(ray, slope, nodes[node_index]))
                {
                    stack_index++;
                    trace_stack[stack_index] = node_index;

                    is_leaf = nodes[node_index].left_leaf_flag;
                    node_index = nodes[node_index].left;

                }
                // If it is not hit, move to the right child of the node at the
                // top of the stack.
                else
                {
                    node_index = trace_stack[stack_index];
                    stack_index--;

                    is_leaf = nodes[node_index].right_leaf_flag;
                    node_index = nodes[node_index].right;
                }
            }

            if (is_leaf && stack_index >= 0)
            {
                // Test sphere inside the current leaf for itersection.
                if (sphere_hit(ray,
                               xs[node_index], ys[node_index], zs[node_index],
                               radii[node_index]))
                {
                    if (ray_hit_count < max_ray_hits)
                        hits[hit_offset+ray_hit_count] = node_index;
                    ray_hit_count++;
                }
                // Move to the right child of the node at the top of the stack.
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
