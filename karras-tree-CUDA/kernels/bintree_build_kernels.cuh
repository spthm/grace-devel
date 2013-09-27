#pragma once

#include <thrust/device_vector.h>

#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

namespace gpu {

/***********************************************/
/**************** CUDA Kenerls. ****************/
/***********************************************/

template <typename UInteger>
__global__ void build_nodes_kernel(Node* nodes,
                                   Leaf* leaves,
                                   UInteger* keys,
                                   unsigned int n_keys,
                                   unsigned char n_bits)
{
    int index, end_index, split_index;
    unsigned int prefix_left, prefix_right, min_prefix, node_prefix;
    unsigned int l_max, l, t, s;
    char direction;

    // Index of this node.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n_keys && index >= 0) {
        // direction == +1 => index is the first key in the node.
        // direction == -1 => index is the last key in the node.
        prefix_left = common_prefix(index, index-1, keys, n_keys, n_bits);
        prefix_right = common_prefix(index, index+1, keys, n_keys, n_bits);
        direction = (prefix_right - prefix_left > 0) ? +1 : -1;

        /* Calculate the index of the other end of the node. */
        // l_max is an upper limit to the distance between the two indices.
        l_max = 2;
        min_prefix = common_prefix(index, index-direction, keys, n_keys, n_bits);
        while (common_prefix(index, index + l_max*direction, keys, n_keys, n_bits)
               > min_prefix) {
            l_max = l_max * 2;
        }
        // Now find l, the distance between the indices, using a binary search.
        // (i.e. one bit at a time.)
        // TODO: Remove l, use end_index only if possible.
        l = 0;
        t = l_max / 2;
        while (t >= 1) {
            if (common_prefix(index, index + (UInteger) (l+t)*direction, keys, n_keys, n_bits)
                > min_prefix) {
                l = l + 1;
            }
        }
        end_index = index + l*direction;

        /* Calculate the index of the split position within the node. */
        node_prefix = common_prefix(index, end_index, keys, n_keys, n_bits);
        s = 0;
        t = (end_index - index) * direction;
        do {
            // t = ceil(t/2.)
            t = (t+1) / 2;
            if (common_prefix(index, index + (s+t)*direction, keys, n_keys, n_bits)
                > node_prefix) {
                s = s + t;
            }
        } while (t > 1);
        // If direction == -1 we actually found split_index + 1;
        split_index = index + s*direction + min(direction, 0);

        /* Update this node with the locations of its children. */
        nodes[index].left = split_index;
        nodes[index].right = split_index+1;
        if (split_index == min(index, end_index)) {
            // Left child is a leaf.
            nodes[index].left_leaf_flag = true;
            leaves[split_index].parent = index;
        }
        else {
            nodes[index].left_leaf_flag = false;
            nodes[split_index].parent = index;
        }

        if (split_index+1 == max(index, end_index)) {
            // Right child is a leaf.
            nodes[index].right_leaf_flag = true;
            leaves[split_index+1].parent = index;
        }
        else {
            nodes[index].right_leaf_flag = false;
            nodes[split_index+1].parent = index;
        }
    }
}

template <typename Float>
__global__ void find_AABBs_kernel(Node* nodes,
                                  Leaf* leaves,
                                  unsigned int n_leaves,
                                  Float* positions,
                                  Float* extent,
                                  unsigned int* AABB_flags)
{
    int index;
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
    float r;
    float *left_bottom, *right_bottom, *left_top, *right_top;

    // Leaf index.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n_leaves && index >= 0) {
        // Find the AABB of each leaf (i.e. each primitive) and write it.
        r = extent[index];
        x_min = positions[index*3 + 0] - r;
        y_min = positions[index*3 + 1] - r;
        z_min = positions[index*3 + 2] - r;

        x_max = x_min + 2*r;
        y_max = y_min + 2*r;
        z_max = z_min + 2*r;

        leaves[index].bottom[0] = x_min;
        leaves[index].bottom[1] = y_min;
        leaves[index].bottom[2] = z_min;

        leaves[index].top[0] = x_max;
        leaves[index].top[1] = y_max;
        leaves[index].top[2] = z_max;

        // Travel up the tree.  The second thread to reach a node writes
        // its AABB based on those of its children.  The first exists the loop.
        index = leaves[index].parent;
        while (true) {
            if (atomicAdd(&AABB_flags[index], 1) == 0) {
                break;
            }
            else {
                if (nodes[index].left_leaf_flag) {
                    left_bottom = leaves[nodes[index].left].bottom;
                    left_top = leaves[nodes[index].left].top;
                }
                else {
                    left_bottom = nodes[nodes[index].left].bottom;
                    left_top = nodes[nodes[index].left].top;
                }
                if (nodes[index].right_leaf_flag) {
                    right_bottom = leaves[nodes[index].right].bottom;
                    right_top = leaves[nodes[index].right].top;
                }
                else {
                    right_bottom = nodes[nodes[index].right].bottom;
                    right_top = nodes[nodes[index].right].top;
                }

                x_min = min(left_bottom[0], right_bottom[0]);
                y_min = min(left_bottom[1], right_bottom[1]);
                z_min = min(left_bottom[2], right_bottom[2]);

                x_max = max(left_top[0], right_top[0]);
                y_max = max(left_top[1], right_top[1]);
                z_max = max(left_top[2], right_top[2]);

                nodes[index].bottom[0] = x_min;
                nodes[index].bottom[1] = y_min;
                nodes[index].bottom[2] = z_min;

                nodes[index].top[0] = x_max;
                nodes[index].top[1] = y_max;
                nodes[index].top[2] = z_max;
            }
            if (index == 0) {
                break;
            }
            index = nodes[index].parent;
        }
    }
}

template <typename UInteger>
__device__ int common_prefix(int i,
                             int j,
                             UInteger *keys,
                             unsigned int n_keys,
                             unsigned char n_bits)
{
    if (i < 0 || i > n_keys || j < 0 || j > n_keys) {
        return -1;
    }
    UInteger key_i = keys[i];
    UInteger key_j = keys[j];

    unsigned int prefix_length = bit_prefix(key_i, key_j);
    if (prefix_length == n_bits) {
        prefix_length += bit_prefix((UInteger)i, (UInteger)j);
    }
    return prefix_length;
}

} // namespace gpu

/**********************************************/
/************** C-like wrappers. **************/
/**********************************************/

template <typename UInteger>
void build_nodes(thrust::device_vector<Node> d_nodes,
                 thrust::device_vector<Leaf> d_leaves,
                 thrust::device_vector<UInteger> d_keys)
{
    unsigned int n_keys = d_keys.size();
    unsigned char n_bits_per_key = CHAR_BIT * sizeof(UInteger);

    gpu::build_nodes_kernel<<<1,1>>>((Node*)thrust::raw_pointer_cast(d_nodes.data()),
                                     (Leaf*)thrust::raw_pointer_cast(d_leaves.data()),
                                     (UInteger*)thrust::raw_pointer_cast(d_keys.data()),
                                     n_keys,
                                     n_bits_per_key);

}

template <typename Float>
void find_AABBs(thrust::device_vector<Node> d_nodes,
                thrust::device_vector<Leaf> d_leaves,
                thrust::device_vector<Float> d_sphere_centres,
                thrust::device_vector<Float> d_sphere_radii)
{
    thrust::device_vector<unsigned int> d_AABB_flags;

    unsigned int n_leaves = d_leaves.size();
    d_AABB_flags.resize(n_leaves);

    gpu::find_AABBs_kernel<<<1,1>>>(thrust::raw_pointer_cast(d_nodes.data()),
                                    thrust::raw_pointer_cast(d_leaves.data()),
                                    n_leaves,
                                    thrust::raw_pointer_cast(d_sphere_centres.data()),
                                    thrust::raw_pointer_cast(d_sphere_radii.data()),
                                    thrust::raw_pointer_cast(d_AABB_flags.data()) );
}


} // namespace grace
