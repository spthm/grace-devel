#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
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
                                   const UInteger* keys,
                                   const UInteger32 n_keys)
{
    Integer32 index, end_index, split_index;
    int prefix_left, prefix_right, min_prefix;
    unsigned int node_prefix;
    unsigned int span_max, l, bit;
    int direction;

    // Index of this node.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < (n_keys-1) && index >= 0)
    {
        // direction == +1 => index is the first key in the node.
        // direction == -1 => index is the last key in the node.
        prefix_left = common_prefix(index, index-1, keys, n_keys);
        prefix_right = common_prefix(index, index+1, keys, n_keys);
        direction = sgn(prefix_right - prefix_left);

        /* Calculate the index of the other end of the node.
         * span_max is an upper limit to the distance between the two indices.
         */
        span_max = 2;
        min_prefix = common_prefix(index, index-direction,
                                   keys, n_keys);
        while (common_prefix(index, index + span_max*direction,
                             keys, n_keys) > min_prefix) {
            span_max = span_max * 2;
        }
        /* Find the distance between the indices, l, using a binary search.
         * (We find each bit t sequantially, starting with the most significant,
         * span_max/2.)
         */
        // TODO: Remove l, use end_index only if possible.
        l = 0;
        bit = span_max / 2;
        while (bit >= 1) {
            if (common_prefix(index, index + (l+bit)*direction,
                              keys, n_keys) > min_prefix) {
                l = l + bit;
            }
            bit = bit / 2;
        }
        end_index = index + l*direction;

        /* Find the index of the split position within the node, l, again
         * using a binary search (see above).
         * In this case, bit could be odd (i.e. not actually one bit), but the
         * principle is the same.
         */
        node_prefix = common_prefix(index, end_index, keys, n_keys);
        bit = l;
        l = 0;
        do {
            // bit = ceil(bit/2.) in case l odd.
            bit = (bit+1) / 2;
            if (common_prefix(index, index + (l+bit)*direction,
                              keys, n_keys) > node_prefix) {
                l = l + bit;
            }
        } while (bit > 1);
        // If direction == -1 we actually found split_index + 1;
        split_index = index + l*direction + min(direction, 0);

        /* Update this node with the locations of its children. */
        nodes[index].level = node_prefix;
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

        index = index + blockDim.x * gridDim.x;
    }
    return;
}

template <typename Float>
__global__ void find_AABBs_kernel(Node* nodes,
                                  Leaf* leaves,
                                  const UInteger32 n_leaves,
                                  const Float* xs,
                                  const Float* ys,
                                  const Float* zs,
                                  const Float* radii,
                                  unsigned int* AABB_flags)
{
    Integer32 tid, index, left_index, right_index;
    Float x_min, y_min, z_min;
    Float x_max, y_max, z_max;
    Float r;
    Float *left_bottom, *right_bottom, *left_top, *right_top;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Leaf index.
    index = tid;
    while (index < n_leaves)
    {
        // Find the AABB of each leaf (i.e. each primitive) and write it.
        r = radii[index];
        x_min = xs[index] - r;
        y_min = ys[index] - r;
        z_min = zs[index] - r;

        x_max = x_min + 2*r;
        y_max = y_min + 2*r;
        z_max = z_min + 2*r;

        leaves[index].bottom[0] = x_min;
        leaves[index].bottom[1] = y_min;
        leaves[index].bottom[2] = z_min;

        leaves[index].top[0] = x_max;
        leaves[index].top[1] = y_max;
        leaves[index].top[2] = z_max;

        __threadfence();

        // Travel up the tree.  The second thread to reach a node writes
        // its AABB based on those of its children.  The first exits the loop.
        index = leaves[index].parent;
        first_arrival = (atomicAdd(&AABB_flags[index], 1) == 0);
        while (true)
        {
            if (first_arrival) {
                break;
            }
            else {
                left_index = nodes[index].left;
                right_index = nodes[index].right;
                if (nodes[index].left_leaf_flag) {
                    left_bottom = leaves[left_index].bottom;
                    left_top = leaves[left_index].top;
                }
                else {
                    left_bottom = nodes[left_index].bottom;
                    left_top = nodes[left_index].top;
                }
                if (nodes[index].right_leaf_flag) {
                    right_bottom = leaves[right_index].bottom;
                    right_top = leaves[right_index].top;
                }
                else {
                    right_bottom = nodes[right_index].bottom;
                    right_top = nodes[right_index].top;
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

                __threadfence();

                // If index == 0 then nodes[index].parent == 0.
                index = nodes[index].parent;
                first_arrival = (atomicAdd(&AABB_flags[index], 1) == 0);
            }
            if (index == 0 && AABB_flags[index] > 2) {
                // If we get to here then the root node has just been processed,
                // which means ALL other nodes have also been processed.
                return;
            }
        }
        tid = tid + blockDim.x * gridDim.x;
        index = tid;
    }
    return;
}

// TODO: Rename this to e.g. common_prefix_length
template <typename UInteger>
__device__ int common_prefix(const Integer32 i,
                             const Integer32 j,
                             const UInteger* keys,
                             const UInteger32 n_keys)
{
    // Should be optimized away by the compiler.
    const unsigned char n_bits = CHAR_BIT * sizeof(UInteger);

    if (j < 0 || j >= n_keys || i < 0 || i >= n_keys) {
        return -1;
    }
    UInteger key_i = keys[i];
    UInteger key_j = keys[j];

    int prefix_length = bit_prefix(key_i, key_j);
    if (prefix_length == n_bits) {
        prefix_length += bit_prefix((UInteger32)i, (UInteger32)j);
    }
    return prefix_length;
}

} // namespace gpu

/**********************************************/
/************** C-like wrappers. **************/
/**********************************************/

template <typename UInteger>
void build_nodes(thrust::device_vector<Node>& d_nodes,
                 thrust::device_vector<Leaf>& d_leaves,
                 const thrust::device_vector<UInteger>& d_keys)
{
    UInteger32 n_keys = d_keys.size();

    int blocks = min(MAX_BLOCKS, (n_keys + THREADS_PER_BLOCK-1)
                                  / THREADS_PER_BLOCK);

    gpu::build_nodes_kernel<<<blocks,THREADS_PER_BLOCK>>>(
        (Node*)thrust::raw_pointer_cast(d_nodes.data()),
        (Leaf*)thrust::raw_pointer_cast(d_leaves.data()),
        (UInteger*)thrust::raw_pointer_cast(d_keys.data()),
        n_keys
    );

}

template <typename Float>
void find_AABBs(thrust::device_vector<Node>& d_nodes,
                thrust::device_vector<Leaf>& d_leaves,
                const thrust::device_vector<Float>& d_sphere_xs,
                const thrust::device_vector<Float>& d_sphere_ys,
                const thrust::device_vector<Float>& d_sphere_zs,
                const thrust::device_vector<Float>& d_sphere_radii)
{
    thrust::device_vector<unsigned int> d_AABB_flags;

    UInteger32 n_leaves = d_leaves.size();
    d_AABB_flags.resize(n_leaves-1);

    int blocks = min(MAX_BLOCKS, (n_leaves + THREADS_PER_BLOCK-1)
                                  / THREADS_PER_BLOCK);

    gpu::find_AABBs_kernel<<<blocks,THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_leaves.data()),
        n_leaves,
        thrust::raw_pointer_cast(d_sphere_xs.data()),
        thrust::raw_pointer_cast(d_sphere_ys.data()),
        thrust::raw_pointer_cast(d_sphere_zs.data()),
        thrust::raw_pointer_cast(d_sphere_radii.data()),
        thrust::raw_pointer_cast(d_AABB_flags.data())
    );
}


} // namespace grace
