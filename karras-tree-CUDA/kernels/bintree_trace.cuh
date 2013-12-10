#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"

namespace grace {

namespace gpu{

template <typename Float>
__global__ void trace_kernel(const Ray* rays,
                             const int n_rays,
                             const int max_ray_hits,
                             int* hits,
                             const Node* nodes,
                             const Leaf* leaves,
                             const Float* xs,
                             const Float* ys,
                             const Float* zs,
                             const Float* radii)
{
    int ray_index, stack_index, hit_offset, hit_count;
    Integer32 node_index;
    bool is_leaf;
    Ray ray;
    // N levels => N-1 split levels (i.e. N-1 key length).
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
        hit_count = 0;
        ray = rays[ray_index];

        while (stack_index >= 0)
        {
            if (!is_leaf)
            {
                if (AABB_hit(ray, nodes[node_index])) {
                    stack_index++;
                    trace_stack[stack_index] = node_index;
                    is_leaf = nodes[node_index].left_leaf_flag;
                    node_index = nodes[nodex_index].left;

                }
                else {
                    node_index = trace_stack[stack_index].right;
                    stack_index--;
                    is_leaf = nodes[node_index].right_leaf_flag;
                    node_index = nodes[node_index].right;
                }
            }

            if (is_leaf)
            {
                if (sphere_hit(ray, leaves[node_index])) {
                    hits[hit_offset+hit_count] = node_index;
                    hit_count++;
                }
                node_index = trace_stack[stack_index];
                stack_index--;
                is_leaf = nodes[node_index].right_leaf_flag;
                node_index = nodes[node_index].right;
            }
        }
        ray_index += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

} // namespace grace
