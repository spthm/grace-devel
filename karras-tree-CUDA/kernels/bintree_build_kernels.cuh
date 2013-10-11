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
                                   const unsigned int n_keys)
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
        direction = (prefix_right - prefix_left > 0) ? +1 : -1;

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
                                  const Vector3<Float>* centres,
                                  const Float* radii,
                                  unsigned int* AABB_flags)
{
    Integer32 index, left_index, right_index;
    Float x_min, y_min, z_min;
    Float x_max, y_max, z_max;
    Float r;
    Float *left_bottom, *right_bottom, *left_top, *right_top;

    // Leaf index.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < n_leaves)
    {
        // Find the AABB of each leaf (i.e. each primitive) and write it.
        r = radii[index];
        x_min = centres[index].x - r;
        y_min = centres[index].y - r;
        z_min = centres[index].z - r;

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
        while (true)
        {
            if (atomicAdd(&AABB_flags[index], 1) == 0) {
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
            }
            if (index == 0) {
                // If we get to here then the root node has just been processed,
                // which means ALL other nodes have also been processed.
                return;
            }
            index = nodes[index].parent;
        }
        index = index + blockDim.x * gridDim.x;
    }
    return;
}

// TODO: Rename this to e.g. common_prefix_length
template <typename UInteger>
__device__ int common_prefix(const Integer32 i,
                             const Integer32 j,
                             const UInteger* keys,
                             const unsigned int n_keys)
{
    // Should be optimized away by the compiler.
    const unsigned char n_bits = CHAR_BIT * sizeof(UInteger);

    if (i < 0 || i >= n_keys || j < 0 || j >= n_keys) {
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

    int threads_per_block = 512;
    int max_blocks = 112; // 7MPs * 16 blocks/MP for compute capability 3.0.
    int blocks = min(max_blocks, (n_keys + threads_per_block-1)
                                  / threads_per_block);

    gpu::build_nodes_kernel<<<blocks,threads_per_block>>>(
        (Node*)thrust::raw_pointer_cast(d_nodes.data()),
        (Leaf*)thrust::raw_pointer_cast(d_leaves.data()),
        (UInteger*)thrust::raw_pointer_cast(d_keys.data()),
        n_keys
    );

}

template <typename Float>
void find_AABBs(thrust::device_vector<Node>& d_nodes,
                thrust::device_vector<Leaf>& d_leaves,
                const thrust::device_vector<Vector3<Float> >& d_sphere_centres,
                const thrust::device_vector<Float>& d_sphere_radii)
{
    thrust::device_vector<unsigned int> d_AABB_flags;

    UInteger32 n_leaves = d_leaves.size();
    d_AABB_flags.resize(n_leaves-1);

    int threads_per_block = 512;
    int max_blocks = 112; // 7MPs * 16 blocks/MP for compute capability 3.0.
    int blocks = min(max_blocks, (n_leaves + threads_per_block-1)
                                  / threads_per_block);

    gpu::find_AABBs_kernel<<<blocks,threads_per_block>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_leaves.data()),
        n_leaves,
        thrust::raw_pointer_cast(d_sphere_centres.data()),
        thrust::raw_pointer_cast(d_sphere_radii.data()),
        thrust::raw_pointer_cast(d_AABB_flags.data())
    );
}


} // namespace grace
