#pragma once

#include <thrust/device_vector.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

namespace gpu {

//-----------------------------------------------------------------------------
// Helper functions for tree build kernels.
//-----------------------------------------------------------------------------

// __device__ and in namespace gpu so we use the __device__ bit_prefix_length()
template <typename UInteger>
__device__ int common_prefix_length(const int i,
                                    const int j,
                                    const UInteger* keys,
                                    const size_t n_keys)
{
    // Should be optimized away by the compiler.
    const unsigned char n_bits = CHAR_BIT * sizeof(UInteger);

    if (j < 0 || j >= n_keys || i < 0 || i >= n_keys) {
        return -1;
    }
    UInteger key_i = keys[i];
    UInteger key_j = keys[j];

    int prefix_length = bit_prefix_length(key_i, key_j);
    if (prefix_length == n_bits) {
        prefix_length += bit_prefix_length((uinteger32)i, (uinteger32)j);
    }
    return prefix_length;
}

//-----------------------------------------------------------------------------
// CUDA kernels for tree building.
//-----------------------------------------------------------------------------

template <typename UInteger>
__global__ void build_nodes_kernel(int4* nodes,
                                   unsigned int* node_levels,
                                   int* leaf_parents,
                                   const UInteger* keys,
                                   const size_t n_keys)
{
    int index, end_index, split_index, direction;
    int prefix_left, prefix_right, min_prefix;
    unsigned int node_prefix;
    unsigned int span_max, l, bit;

    // Index of the current node.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < (n_keys-1) && index >= 0)
    {
        prefix_left = common_prefix_length(index, index-1, keys, n_keys);
        prefix_right = common_prefix_length(index, index+1, keys, n_keys);
        // direction == +1 => index is the first key in the node.
        //              -1 => index is the last key in the node.
        direction = sgn(prefix_right - prefix_left);

        // Calculate an upper limit to the size of the current node (the number
        // of keys it spans).
        span_max = 2;
        min_prefix = common_prefix_length(index, index-direction,
                                          keys, n_keys);
        while (common_prefix_length(index, index + span_max*direction,
                                    keys, n_keys) > min_prefix) {
            span_max = span_max * 2;
        }

        // Perform a binary search for the other end of the node, beginning
        // with the upper limit from above.
        l = 0;
        bit = span_max / 2;
        while (bit >= 1) {
            if (common_prefix_length(index, index + (l+bit)*direction,
                                     keys, n_keys) > min_prefix) {
                l = l + bit;
            }
            bit = bit / 2;
        }
        end_index = index + l*direction;

        // Perform a binary search for the node's split position.
        node_prefix = common_prefix_length(index, end_index, keys, n_keys);
        bit = l;
        l = 0;
        do {
            // bit = ceil(bit/2.0) in case bit odd.
            bit = (bit+1) / 2;
            if (common_prefix_length(index, index + (l+bit)*direction,
                                     keys, n_keys) > node_prefix) {
                l = l + bit;
            }
        } while (bit > 1);
        // If direction == -1 we actually found split_index + 1.
        split_index = index + l*direction + min(direction, 0);

        node_levels[index] = node_prefix;
        nodes[index].x = split_index;
        nodes[index].y = split_index+1;
        nodes[index].w = end_index;

        // Leaves are identified by their indicies, which are >= n_nodes
        // (and n_nodes == n_keys-1).
        if (split_index == min(index, end_index)) {
            nodes[index].x += n_keys-1;
            leaf_parents[split_index] = index;
        }
        else {
            nodes[split_index].z = index;
        }

        if (split_index+1 == max(index, end_index)) {
            nodes[index].y += n_keys-1;
            leaf_parents[split_index+1] = index;
        }
        else {
            nodes[split_index+1].z = index;
        }

        index += blockDim.x * gridDim.x;
    }
    return;
}

template <typename Float, typename Float4>
__global__ void find_AABBs_kernel(const int4* nodes,
                                  volatile Box* node_AABBs,
                                  const int* leaf_parents,
                                  volatile Box* leaf_AABBs,
                                  const size_t n_leaves,
                                  const Float4* spheres,
                                  unsigned int* g_flags)
{
    int tid, index, flag_index, block_lower, block_upper;
    int4 node;
    Float x_min, y_min, z_min;
    Float x_max, y_max, z_max;
    Float r;
    volatile Box *left_AABB, *right_AABB;
    unsigned int* flags;
    bool first_arrival, in_block;

    // Use shared memory for the N-accessed flags when all children of a node
    // have been processed in the same block.
    __shared__ unsigned int sm_flags[AABB_THREADS_PER_BLOCK];
    sm_flags[threadIdx.x] = 0;
    __syncthreads();
    block_lower = blockIdx.x * AABB_THREADS_PER_BLOCK;
    block_upper = block_lower + AABB_THREADS_PER_BLOCK - 1;

    tid = threadIdx.x + blockIdx.x * AABB_THREADS_PER_BLOCK;

    // Loop provided there are > 0 threads in this block with tid < n_leaves,
    // so all threads hit the __syncthreads().
    while (tid - threadIdx.x < n_leaves)
    {
        if (tid < n_leaves)
        {
            Float4 sphere = spheres[tid];
            r = sphere.w;
            x_min = sphere.x - r;
            y_min = sphere.y - r;
            z_min = sphere.z - r;

            x_max = x_min + 2*r;
            y_max = y_min + 2*r;
            z_max = z_min + 2*r;

            leaf_AABBs[tid].tx = x_max;
            leaf_AABBs[tid].ty = y_max;
            leaf_AABBs[tid].tz = z_max;

            leaf_AABBs[tid].bx = x_min;
            leaf_AABBs[tid].by = y_min;
            leaf_AABBs[tid].bz = z_min;

            // Travel up the tree.  The second thread to reach a node writes
            // its AABB based on those of its children.  The first exits the
            // loop.
            index = leaf_parents[tid];
            node = nodes[index]; // Left, right, parent, end.
            in_block = (min(node.w, index) >= block_lower &&
                        max(node.w, index) <= block_upper);

            if (in_block) {
                flags = sm_flags;
                flag_index = index % AABB_THREADS_PER_BLOCK;
                __threadfence_block();
            }
            else {
                flags = g_flags;
                flag_index = index;
                __threadfence();
            }

            first_arrival = (atomicAdd(&flags[flag_index], 1) == 0);
            while (!first_arrival)
            {
                if (node.x > n_leaves-2)
                    left_AABB = &leaf_AABBs[(node.x - (n_leaves-1))];
                else
                    left_AABB = &node_AABBs[node.x];

                if (node.y > n_leaves-2)
                    right_AABB = &leaf_AABBs[(node.y - (n_leaves-1))];
                else
                    right_AABB = &node_AABBs[node.y];


                x_max = max(left_AABB->tx, right_AABB->tx);
                y_max = max(left_AABB->ty, right_AABB->ty);
                z_max = max(left_AABB->tz, right_AABB->tz);

                x_min = min(left_AABB->bx, right_AABB->bx);
                y_min = min(left_AABB->by, right_AABB->by);
                z_min = min(left_AABB->bz, right_AABB->bz);

                node_AABBs[index].tx = x_max;
                node_AABBs[index].ty = y_max;
                node_AABBs[index].tz = z_max;

                node_AABBs[index].bx = x_min;
                node_AABBs[index].by = y_min;
                node_AABBs[index].bz = z_min;

                if (index == 0) {
                    // Root node processed, so all nodes processed.
                    // Break rather than return because of the __syncthreads();
                    break;
                }

                index = node.z;
                node = nodes[index];
                in_block = (min(node.w, index) >= block_lower &&
                            max(node.w, index) <= block_upper);

                if (in_block) {
                    flags = sm_flags;
                    flag_index = index % AABB_THREADS_PER_BLOCK;
                    __threadfence_block();
                }
                else {
                    flags = g_flags;
                    flag_index = index;
                    __threadfence();
                }

                first_arrival = (atomicAdd(&flags[flag_index], 1) == 0);
            }
        }
        // Before we move on to a new block of leaves to process, wipe shared
        // memory flags so all threads agree what sm_flags[i] corresponds to.
        __syncthreads();
        sm_flags[threadIdx.x] = 0;
        __syncthreads();

        tid += AABB_THREADS_PER_BLOCK * gridDim.x;
        block_lower += AABB_THREADS_PER_BLOCK * gridDim.x;
        block_upper += AABB_THREADS_PER_BLOCK * gridDim.x;
    }
    return;
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tree building.
//-----------------------------------------------------------------------------

template <typename UInteger>
void build_nodes(Nodes& d_nodes,
                 Leaves& d_leaves,
                 const thrust::device_vector<UInteger>& d_keys)
{
    size_t n_keys = d_keys.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_keys + BUILD_THREADS_PER_BLOCK-1)
                                        / BUILD_THREADS_PER_BLOCK));

    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.

    gpu::build_nodes_kernel<<<blocks,BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.level.data()),
        thrust::raw_pointer_cast(d_leaves.parent.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        n_keys);
}

template <typename Float4>
void find_AABBs(Nodes& d_nodes,
                Leaves& d_leaves,
                const thrust::device_vector<Float4>& d_spheres)
{
    size_t n_leaves = d_leaves.parent.size();

    thrust::device_vector<unsigned int> d_AABB_flags(n_leaves-1);

    int blocks = min(MAX_BLOCKS, (int) ((n_leaves + AABB_THREADS_PER_BLOCK-1)
                                        / AABB_THREADS_PER_BLOCK));

    gpu::find_AABBs_kernel<float, Float4><<<blocks,AABB_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        thrust::raw_pointer_cast(d_leaves.parent.data()),
        thrust::raw_pointer_cast(d_leaves.AABB.data()),
        n_leaves,
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_AABB_flags.data())
    );
}


} // namespace grace
