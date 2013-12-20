#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"
#include "../ray.h"

namespace grace {

enum CLASSIFICATION
{ MMM, PMM, MPM, PPM, MMP, PMP, MPP, PPP };

__host__ __device__ bool AABB_hit_eisemann(const Ray& ray, const Node& node)
{
    float ox = ray.ox;
    float oy = ray.oy;
    float oz = ray.oz;

    float xbyy = ray.xbyy;
    float ybyx = ray.ybyx;
    float ybyz = ray.ybyz;
    float zbyy = ray.zbyy;
    float xbyz = ray.xbyz;
    float zbyx = ray.zbyx;

    float c_xy = ray.c_zy;
    float c_xz = ray.c_zy;
    float c_yx = ray.c_zy;
    float c_yz = ray.c_zy;
    float c_zx = ray.c_zy;
    float c_zy = ray.c_zy;

    float bx = node.bottom[0];
    float by = node.bottom[1];
    float bz = node.bottom[2];
    float tx = node.top[0];
    float ty = node.top[1];
    float tz = node.top[2];

    switch(ray.dclass)
    {
    case MMM:

        if ((ox < bx) || (oy < by) || (oz < bz)
            || (ybyx * bx - ty + c_xy > 0)
            || (xbyy * by - tx + c_yx > 0)
            || (ybyz * bz - ty + c_zy > 0)
            || (zbyy * by - tz + c_yz > 0)
            || (zbyx * bx - tz + c_xz > 0)
            || (xbyz * bz - tx + c_zx > 0)
            )
            return false;

        return true;

    case MMP:

        if ((ox < bx) || (oy < by) || (oz > tz)
            || (ybyx * bx - ty + c_xy > 0)
            || (xbyy * by - tx + c_yx > 0)
            || (ybyz * tz - ty + c_zy > 0)
            || (zbyy * by - bz + c_yz < 0)
            || (zbyx * bx - bz + c_xz < 0)
            || (xbyz * tz - tx + c_zx > 0)
            )
            return false;

        return true;

    case MPM:

        if ((ox < bx) || (oy > ty) || (oz < bz)
            || (ybyx * bx - by + c_xy < 0)
            || (xbyy * ty - tx + c_yx > 0)
            || (ybyz * bz - by + c_zy < 0)
            || (zbyy * ty - tz + c_yz > 0)
            || (zbyx * bx - tz + c_xz > 0)
            || (xbyz * bz - tx + c_zx > 0)
            )
            return false;

        return true;

    case MPP:

        if ((ox < bx) || (oy > ty) || (oz > tz)
            || (ybyx * bx - by + c_xy < 0)
            || (xbyy * ty - tx + c_yx > 0)
            || (ybyz * tz - by + c_zy < 0)
            || (zbyy * ty - bz + c_yz < 0)
            || (zbyx * bx - bz + c_xz < 0)
            || (xbyz * tz - tx + c_zx > 0)
            )
            return false;

        return true;

    case PMM:

        if ((ox > tx) || (oy < by) || (oz < bz)
            || (ybyx * tx - ty + c_xy > 0)
            || (xbyy * by - bx + c_yx < 0)
            || (ybyz * bz - ty + c_zy > 0)
            || (zbyy * by - tz + c_yz > 0)
            || (zbyx * tx - tz + c_xz > 0)
            || (xbyz * bz - bx + c_zx < 0)
            )
            return false;

        return true;

    case PMP:

        if ((ox > tx) || (oy < by) || (oz > tz)
            || (ybyx * tx - ty + c_xy > 0)
            || (xbyy * by - bx + c_yx < 0)
            || (ybyz * tz - ty + c_zy > 0)
            || (zbyy * by - bz + c_yz < 0)
            || (zbyx * tx - bz + c_xz < 0)
            || (xbyz * tz - bx + c_zx < 0)
            )
            return false;

        return true;

    case PPM:

        if ((ox > tx) || (oy > ty) || (oz < bz)
            || (ybyx * tx - by + c_xy < 0)
            || (xbyy * ty - bx + c_yx < 0)
            || (ybyz * bz - by + c_zy < 0)
            || (zbyy * ty - tz + c_yz > 0)
            || (zbyx * tx - tz + c_xz > 0)
            || (xbyz * bz - bx + c_zx < 0)
            )
            return false;

        return true;

    case PPP:

        if ((ox > tx) || (oy > ty) || (oz > tz)
            || (ybyx * tx - by + c_xy < 0)
            || (xbyy * ty - bx + c_yx < 0)
            || (ybyz * tz - by + c_zy < 0)
            || (zbyy * ty - bz + c_yz < 0)
            || (zbyx * tx - bz + c_xz < 0)
            || (xbyz * tz - bx + c_zx < 0)
            )
            return false;

        return true;

    // case OMM:

    //     if((ox < bx) || (ox > tx)
    //         || (oy < by) || (oz < bz)
    //         || (ybyz * bz - ty + c_zy > 0)
    //         || (zbyy * by - tz + c_yz > 0)
    //         )
    //         return false;

    //     return true;

    // case OMP:

    //     if((ox < bx) || (ox > tx)
    //         || (oy < by) || (oz > tz)
    //         || (ybyz * tz - ty + c_zy > 0)
    //         || (zbyy * by - bz + c_yz < 0)
    //         )
    //         return false;

    //     return true;

    // case OPM:

    //     if((ox < bx) || (ox > tx)
    //         || (oy > ty) || (oz < bz)
    //         || (ybyz * bz - by + c_zy < 0)
    //         || (zbyy * ty - tz + c_yz > 0)
    //         )
    //         return false;

    //     return true;

    // case OPP:

    //     if((ox < bx) || (ox > tx)
    //         || (oy > ty) || (oz > tz)
    //         || (ybyz * tz - by + c_zy < 0)
    //         || (zbyy * ty - bz + c_yz < 0)
    //         )
    //         return false;

    //     return true;

    // case MOM:

    //     if((oy < by) || (oy > ty)
    //         || (ox < bx) || (oz < bz)
    //         || (zbyx * bx - tz + c_xz > 0)
    //         || (xbyz * bz - tx + c_zx > 0)
    //         )
    //         return false;

    //     return true;

    // case MOP:

    //     if((oy < by) || (oy > ty)
    //         || (ox < bx) || (oz > tz)
    //         || (zbyx * bx - bz + c_xz < 0)
    //         || (xbyz * tz - tx + c_zx > 0)
    //         )
    //         return false;

    //     return true;

    // case POM:

    //     if((oy < by) || (oy > ty)
    //         || (ox > tx) || (oz < bz)
    //         || (zbyx * tx - tz + c_xz > 0)
    //         || (xbyz * bz - bx + c_zx < 0)
    //         )
    //         return false;

    //     return true;

    // case POP:

    //     if((oy < by) || (oy > ty)
    //         || (ox > tx) || (oz > tz)
    //         || (zbyx * tx - bz + c_xz < 0)
    //         || (xbyz * tz - bx + c_zx < 0)
    //         )
    //         return false;

    //     return true;

    // case MMO:

    //     if((oz < bz) || (oz > tz)
    //         || (ox < bx) || (oy < by)
    //         || (ybyx * bx - ty + c_xy > 0)
    //         || (xbyy * by - tx + c_yx > 0)
    //         )
    //         return false;

    //     return true;

    // case MPO:

    //     if((oz < bz) || (oz > tz)
    //         || (ox < bx) || (oy > ty)
    //         || (ybyx * bx - by + c_xy < 0)
    //         || (xbyy * ty - tx + c_yx > 0)
    //         )
    //         return false;

    //     return true;

    // case PMO:

    //     if((oz < bz) || (oz > tz)
    //         || (ox > tx) || (oy < by)
    //         || (ybyx * tx - ty + c_xy > 0)
    //         || (xbyy * by - bx + c_yx < 0)
    //         )
    //         return false;

    //     return true;

    // case PPO:

    //     if((oz < bz) || (oz > tz)
    //         || (ox > tx) || (oy > ty)
    //         || (ybyx * tx - by + c_xy < 0)
    //         || (xbyy * ty - bx + c_yx < 0)
    //         )
    //         return false;

    //     return true;

    // case MOO:

    //     if((ox < bx)
    //         || (oy < by) || (oy > ty)
    //         || (oz < bz) || (oz > tz)
    //         )
    //         return false;

    //     return true;

    // case POO:

    //     if((ox > tx)
    //         || (oy < by) || (oy > ty)
    //         || (oz < bz) || (oz > tz)
    //         )
    //         return false;

    //     return true;

    // case OMO:

    //     if((oy < by)
    //         || (ox < bx) || (ox > tx)
    //         || (oz < bz) || (oz > tz)
    //         )
    //         return false;

    // case OPO:

    //     if((oy > ty)
    //         || (ox < bx) || (ox > tx)
    //         || (oz < bz) || (oz > tz)
    //         )
    //         return false;

    // case OOM:

    //     if((oz < bz)
    //         || (ox < bx) || (ox > tx)
    //         || (oy < by) || (oy > ty)
    //         )
    //         return false;

    // case OOP:

    //     if((oz > tz)
    //         || (ox < bx) || (ox > tx)
    //         || (oy < by) || (oy > ty)
    //         )
    //         return false;

    //     return true;

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
    // N (31) levels => N-1 (30) key length.
    // One extra so we can avoid stack_index = -1 before trace exit.
    __shared__ int trace_stack[31*TRACE_THREADS_PER_BLOCK];

    ray_index = threadIdx.x + blockIdx.x * blockDim.x;
    // Top of the stack.  Must provide a valid node index, so points to root.
    trace_stack[threadIdx.x*31] = 0;

    while (ray_index <= n_rays)
    {
        node_index = 0;
        stack_index = threadIdx.x*31;
        is_leaf = false;

        hit_offset = ray_index*max_ray_hits;
        ray_hit_count = 0;

        while (stack_index >= (int) threadIdx.x*31)
        {
            if (!is_leaf)
            {
                if (AABB_hit_eisemann(rays[ray_index], nodes[node_index])) {
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
                if (sphere_hit(rays[ray_index],
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
