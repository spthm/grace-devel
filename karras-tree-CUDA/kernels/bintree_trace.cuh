#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"
#include "../ray.h"

namespace grace {

__host__ __device__ bool AABB_hit(const Ray& ray, const Node& node) {
    float dx = ray.dx;
    float dy = ray.dy;
    float dz = ray.dz;
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

    e2bx = s2bx - dx*length;
    e2by = s2by - dy*length;
    e2bz = s2bz - dz*length;

    e2tx = s2tx - dx*length;
    e2ty = s2ty - dy*length;
    e2tz = s2tz - dz*length;

    switch(ray.dclass)
    {
        // MMM
        case 0:
        if (s2bx > 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2ty < 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (dx*s2by - dy*s2tx < 0.0f ||
            dx*s2ty - dy*s2bx > 0.0f ||
            dx*s2tz - dz*s2bx > 0.0f ||
            dx*s2bz - dz*s2tx < 0.0f ||
            dy*s2bz - dz*s2ty < 0.0f ||
            dy*s2tz - dz*s2by > 0.0f ) return false;
        break;

        // PMM
        case 1:
        if (s2tx < 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2ty < 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (dx*s2ty - dy*s2tx < 0.0f ||
            dx*s2by - dy*s2bx > 0.0f ||
            dx*s2bz - dz*s2bx > 0.0f ||
            dx*s2tz - dz*s2tx < 0.0f ||
            dy*s2bz - dz*s2ty < 0.0f ||
            dy*s2tz - dz*s2by > 0.0f) return false;
        break;

        // MPM
        case 2:
        if (s2bx > 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2by > 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (dx*s2by - dy*s2bx < 0.0f ||
            dx*s2ty - dy*s2tx > 0.0f ||
            dx*s2tz - dz*s2bx > 0.0f ||
            dx*s2bz - dz*s2tx < 0.0f ||
            dy*s2tz - dz*s2ty < 0.0f ||
            dy*s2bz - dz*s2by > 0.0f) return false;
        break;

        // PPM
        case 3:
        if (s2tx < 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2by > 0.0f || e2tz < 0.0f)
            return false; // past length of ray

        if (dx*s2ty - dy*s2bx < 0.0f ||
            dx*s2by - dy*s2tx > 0.0f ||
            dx*s2bz - dz*s2bx > 0.0f ||
            dx*s2tz - dz*s2tx < 0.0f ||
            dy*s2tz - dz*s2ty < 0.0f ||
            dy*s2bz - dz*s2by > 0.0f) return false;
        break;

        // MMP
        case 4:
        if (s2bx > 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2ty < 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (dx*s2by - dy*s2tx < 0.0f ||
            dx*s2ty - dy*s2bx > 0.0f ||
            dx*s2tz - dz*s2tx > 0.0f ||
            dx*s2bz - dz*s2bx < 0.0f ||
            dy*s2bz - dz*s2by < 0.0f ||
            dy*s2tz - dz*s2ty > 0.0f) return false;
        break;

        // PMP
        case 5:
        if (s2tx < 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2ty < 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (dx*s2ty - dy*s2tx < 0.0f ||
            dx*s2by - dy*s2bx > 0.0f ||
            dx*s2bz - dz*s2tx > 0.0f ||
            dx*s2tz - dz*s2bx < 0.0f ||
            dy*s2bz - dz*s2by < 0.0f ||
            dy*s2tz - dz*s2ty > 0.0f) return false;
        break;

        // MPP
        case 6:
        if (s2bx > 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2tx < 0.0f || e2by > 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (dx*s2by - dy*s2bx < 0.0f ||
            dx*s2ty - dy*s2tx > 0.0f ||
            dx*s2tz - dz*s2tx > 0.0f ||
            dx*s2bz - dz*s2bx < 0.0f ||
            dy*s2tz - dz*s2by < 0.0f ||
            dy*s2bz - dz*s2ty > 0.0f) return false;
        break;

        // PPP
        case 7:
        if (s2tx < 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // on negative part of ray

        if (e2bx > 0.0f || e2by > 0.0f || e2bz > 0.0f)
            return false; // past length of ray

        if (dx*s2ty - dy*s2bx < 0.0f ||
            dx*s2by - dy*s2tx > 0.0f ||
            dx*s2bz - dz*s2tx > 0.0f ||
            dx*s2tz - dz*s2bx < 0.0f ||
            dy*s2tz - dz*s2by < 0.0f ||
            dy*s2bz - dz*s2ty > 0.0f) return false;
        break;
    }
    // Didn't return false above, so we have a hit.
    return true;

}

__device__ bool sphere_hit(Ray ray, Leaf leaf) {
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
                if (sphere_hit(ray, leaves[node_index])) {
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
