#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

namespace gpu {

//-----------------------------------------------------------------------------
// CUDA kernels for tree building
//-----------------------------------------------------------------------------

template <typename UInteger>
__global__ void build_nodes_kernel(integer32* nodes_left,
                                   integer32* nodes_right,
                                   integer32* nodes_parent,
                                   integer32* nodes_end,
                                   unsigned int* nodes_level,
                                   integer32* leaves_parent,
                                   const UInteger* keys,
                                   const size_t n_keys)
{
    integer32 index, end_index, split_index;
    int prefix_left, prefix_right, min_prefix;
    unsigned int node_prefix;
    unsigned int span_max, l, bit;
    int direction;

    // Index of this node.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < (n_keys-1) && index >= 0)
    {
        // direction == +1 => index is the first key in the node.
        //              -1 => index is the last key in the node.
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
         * (We find each bit t sequantially, starting with the most
         * significant, span_max/2.)
         */
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

        nodes_level[index] = node_prefix;
        nodes_left[index] = split_index;
        nodes_right[index] = split_index+1;
        nodes_end[index] = end_index;

        if (split_index == min(index, end_index)) {
            nodes_left[index] += n_keys-1;
            leaves_parent[split_index] = index;
        }
        else {
            nodes_parent[split_index] = index;
        }

        if (split_index+1 == max(index, end_index)) {
            nodes_right[index] += n_keys-1;
            leaves_parent[split_index+1] = index;
        }
        else {
            nodes_parent[split_index+1] = index;
        }

        index += blockDim.x * gridDim.x;
    }
    return;
}

template <typename Float, typename Float4>
__global__ void find_AABBs_kernel(const integer32* nodes_left,
                                  const integer32* nodes_right,
                                  const integer32* nodes_parent,
                                  const integer32* nodes_end,
                                  volatile Box* nodes_AABB,
                                  const integer32* leaves_parent,
                                  volatile Box* leaves_AABB,
                                  const size_t n_leaves,
                                  const Float4* spheres_xyzr,
                                  unsigned int* g_flags)
{
    integer32 tid, index, left_index, right_index;
    integer32 flag_index, block_lower, block_upper;
    Float x_min, y_min, z_min;
    Float x_max, y_max, z_max;
    Float r;
    volatile Box *left_AABB, *right_AABB;
    unsigned int* flags;
    bool first_arrival, in_block;

    __shared__ unsigned int sm_flags[AABB_THREADS_PER_BLOCK];
    block_lower = blockIdx.x * AABB_THREADS_PER_BLOCK;
    block_upper = block_lower + AABB_THREADS_PER_BLOCK - 1;

    tid = threadIdx.x + blockIdx.x * AABB_THREADS_PER_BLOCK;

    // Loop provided there are > 0 threads in this block with tid < n_leaves,
    // so all threads hit the __syncthreads().
    while (tid - threadIdx.x < n_leaves)
    {
        if (tid < n_leaves)
        {
            // Find the AABB of each leaf (i.e. each primitive) and write it.
            Float4 xyzr = spheres_xyzr[tid];
            r = xyzr.w;
            x_min = xyzr.x - r;
            y_min = xyzr.y - r;
            z_min = xyzr.z - r;

            x_max = x_min + 2*r;
            y_max = y_min + 2*r;
            z_max = z_min + 2*r;

            leaves_AABB[tid].tx = x_max;
            leaves_AABB[tid].ty = y_max;
            leaves_AABB[tid].tz = z_max;

            leaves_AABB[tid].bx = x_min;
            leaves_AABB[tid].by = y_min;
            leaves_AABB[tid].bz = z_min;

            // Travel up the tree.  The second thread to reach a node writes
            // its AABB based on those of its children.
            // The first exits the loop.
            index = leaves_parent[tid];
            in_block = (min(nodes_end[index], index) >= block_lower &&
                        max(nodes_end[index], index) <= block_upper);

            flags = sm_flags;
            flag_index = index % AABB_THREADS_PER_BLOCK;
            __threadfence_block();

            if (!in_block) {
                flags = g_flags;
                flag_index = index;
                __threadfence();
            }

            first_arrival = (atomicAdd(&flags[flag_index], 1) == 0);
            while (!first_arrival)
            {
                left_index = nodes_left[index];
                right_index = nodes_right[index];
                if (left_index > n_leaves-2) {
                    left_AABB = &leaves_AABB[(left_index-n_leaves+1)];
                }
                else {
                    left_AABB = &nodes_AABB[left_index];
                }
                if (right_index > n_leaves-2) {
                    right_AABB = &leaves_AABB[(right_index-n_leaves+1)];
                }
                else {
                    right_AABB = &nodes_AABB[right_index];
                }

                x_max = max(left_AABB->tx, right_AABB->tx);
                y_max = max(left_AABB->ty, right_AABB->ty);
                z_max = max(left_AABB->tz, right_AABB->tz);

                x_min = min(left_AABB->bx, right_AABB->bx);
                y_min = min(left_AABB->by, right_AABB->by);
                z_min = min(left_AABB->bz, right_AABB->bz);

                nodes_AABB[index].tx = x_max;
                nodes_AABB[index].ty = y_max;
                nodes_AABB[index].tz = z_max;

                nodes_AABB[index].bx = x_min;
                nodes_AABB[index].by = y_min;
                nodes_AABB[index].bz = z_min;

                if (index == 0) {
                    // Root node processed, so all nodes processed.
                    // Break rather than return because of the __syncthreads();
                    break;
                }

                index = nodes_parent[index];
                in_block = (min(nodes_end[index], index) >= block_lower &&
                            max(nodes_end[index], index) <= block_upper);

                flags = sm_flags;
                flag_index = index % AABB_THREADS_PER_BLOCK;
                __threadfence_block();

                if (!in_block) {
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

// TODO: Rename this to e.g. common_prefix_length
template <typename UInteger>
__device__ int common_prefix(const integer32 i,
                             const integer32 j,
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

    int prefix_length = bit_prefix(key_i, key_j);
    if (prefix_length == n_bits) {
        prefix_length += bit_prefix((uinteger32)i, (uinteger32)j);
    }
    return prefix_length;
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tree building
//-----------------------------------------------------------------------------

template <typename UInteger>
void build_nodes(Nodes& d_nodes, Leaves& d_leaves,
                 const thrust::device_vector<UInteger>& d_keys)
{
    size_t n_keys = d_keys.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_keys + BUILD_THREADS_PER_BLOCK-1)
                                        / BUILD_THREADS_PER_BLOCK));

    // TODO: Error if n_keys <= 1.

    gpu::build_nodes_kernel<<<blocks,BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.left.data()),
        thrust::raw_pointer_cast(d_nodes.right.data()),
        thrust::raw_pointer_cast(d_nodes.parent.data()),
        thrust::raw_pointer_cast(d_nodes.end.data()),
        thrust::raw_pointer_cast(d_nodes.level.data()),
        thrust::raw_pointer_cast(d_leaves.parent.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        n_keys);
}

template <typename Float4>
void find_AABBs(Nodes& d_nodes, Leaves& d_leaves,
                const thrust::device_vector<Float4>& d_spheres_xyzr)
{
    thrust::device_vector<unsigned int> d_AABB_flags;

    size_t n_leaves = d_leaves.parent.size();
    d_AABB_flags.resize(n_leaves-1);

    int blocks = min(MAX_BLOCKS, (int) ((n_leaves + AABB_THREADS_PER_BLOCK-1)
                                        / AABB_THREADS_PER_BLOCK));

    gpu::find_AABBs_kernel<float, Float4><<<blocks,AABB_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.left.data()),
        thrust::raw_pointer_cast(d_nodes.right.data()),
        thrust::raw_pointer_cast(d_nodes.parent.data()),
        thrust::raw_pointer_cast(d_nodes.end.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        thrust::raw_pointer_cast(d_leaves.parent.data()),
        thrust::raw_pointer_cast(d_leaves.AABB.data()),
        n_leaves,
        thrust::raw_pointer_cast(d_spheres_xyzr.data()),
        thrust::raw_pointer_cast(d_AABB_flags.data())
    );
}


} // namespace grace
