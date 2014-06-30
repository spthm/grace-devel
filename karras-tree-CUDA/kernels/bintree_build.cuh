#pragma once

#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {


//-----------------------------------------------------------------------------
// Helper functions for tree build kernels.
//-----------------------------------------------------------------------------

struct flag_null_node
{
    __host__ __device__ int operator() (const int4 node)
    {
        // node.y: index of right child (cannot be the root node)
        // leaf.y: spheres within leaf (cannot be < 1)
        if (node.y > 0)
            return 0;
        else
            return 1;
    }
};

struct is_valid_node
{
    __host__ __device__ bool operator() (const int4 node)
    {
        return (node.y > 0);
    }
};

struct is_valid_level
{
    __host__ __device__ bool operator() (const unsigned int level)
    {
        return (level != 0);
    }
};

namespace gpu {

//-----------------------------------------------------------------------------
// GPU helper functions for tree build kernels.
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

void copy_valid_nodes(thrust::device_vector<int4>& d_nodes,
                      const size_t N_nodes)
{
    thrust::device_vector<int4> d_tmp = d_nodes;
    d_nodes.resize(N_nodes);
    thrust::copy_if(d_tmp.begin(), d_tmp.end(),
                    d_nodes.begin(),
                    is_valid_node());
}

void copy_valid_levels(thrust::device_vector<unsigned int>& d_levels,
                       const size_t N_nodes)
{
    thrust::device_vector<unsigned int> d_tmp = d_levels;
    d_levels.resize(N_nodes);
    thrust::copy_if(d_tmp.begin()+1, d_tmp.end(),
                    d_levels.begin()+1,
                    is_valid_level());
}

//-----------------------------------------------------------------------------
// CUDA kernels for tree building.
//-----------------------------------------------------------------------------

template <typename UInteger>
__global__ void build_nodes_kernel(int4* nodes,
                                   unsigned int* node_levels,
                                   int4* leaves,
                                   const unsigned int max_per_leaf,
                                   const UInteger* keys,
                                   const size_t n_keys)
{
    int index, end_index, split_index, direction;
    int prefix_left, prefix_right, min_prefix;
    unsigned int node_prefix;
    unsigned int span_max, l, bit;
    int4 left, right;

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

        // Check we have a valid node, i.e. its span is > max_per_leaf.
        if (abs(end_index - index) + 1 > max_per_leaf)
        {
            node_levels[index] = node_prefix;
            nodes[index].x = split_index;
            nodes[index].y = split_index+1;
            nodes[index].w = end_index - index;

            left.x = min(index, end_index); // start
            left.y = (split_index - left.x) + 1; // primitives count
            right.x = left.x + left.y;
            right.y = max(index, end_index) - split_index;

            assert(left.y > 0);
            assert(right.y > 0);
            assert(right.x + right.y - 1 < n_keys);

            // Leaves are identified by their indicies, which are >= n_nodes
            // (and currently n_nodes == n_keys-1).
            if (left.y <= max_per_leaf) {
                // Left child is a leaf.
                nodes[index].x += n_keys-1;
                left.z = index;
                leaves[split_index] = left;
            }
            else {
                // Left child is a node.
                nodes[split_index].z = index;
            }

            if (right.y <= max_per_leaf) {
                nodes[index].y += n_keys-1;
                right.z = index;
                leaves[split_index+1] = right;
            }
            else {
                nodes[split_index+1].z = index;
            }
        } // No else.  We do not write to an invalid node.

        index += blockDim.x * gridDim.x;
    }
    return;
}

__global__ void shift_tree_indices(int4* nodes,
                                   int4* leaves,
                                   const unsigned int* leaf_shifts,
                                   const unsigned int n_removed,
                                   const size_t n_nodes)
{
    int4 node;
    int tid;
    unsigned int shift;
    size_t n_nodes_prior;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    n_nodes_prior = n_nodes + n_removed;

    while (tid < n_nodes)
    {
        node = nodes[tid];

        if (node.x >= n_nodes_prior) {
            // A leaf is identified by an index >= n_nodes.  Since n_nodes has
            // been reduced, as additional shift is required.
            shift = leaf_shifts[node.x-n_nodes_prior] + n_removed;
            assert(node.x-shift >= n_nodes);
        }
        else {
            shift = leaf_shifts[node.x];
            assert(node.x-shift < n_nodes);
        }
        node.x -= shift;

        if (node.y >= n_nodes_prior) {
            shift = leaf_shifts[node.y-n_nodes_prior] + n_removed;
            assert(node.y-shift >= n_nodes);
        }
        else {
            // For a right node, we use the shift for its left sibling.
            // (right_index-1 == left_index)
            shift = leaf_shifts[node.y-1];
            assert(node.y-shift < n_nodes);
        }
        node.y -= shift;

        nodes[tid].x = node.x;
        nodes[tid].y = node.y;

        // Do this near the top, when we read nodes[tid]?  May give slightly
        // more coalesced memory accesses.
        // Don't forget to change >= n_nodes to >= n_nodes_prior if moved!
        // TODO: Wrap the asserts in an #ifndef NDEBUG, define some local
        //       variables and tidy up the code.
        if (node.x >= n_nodes) {
            // Current node can only have shifted by some distance >= 0.
            assert(tid <= leaves[node.x-n_nodes].z);
            // Current node cannot have shifted  by any distance > n_removed.
            assert(leaves[node.x-n_nodes].z - tid <= n_removed);
            // We do not know if the current node is a left or a right child,
            // so OR the conditions for both possibilities, respectively.
            // The root node is technically a right child, but cannot be shifted
            // so is a special case.
            assert(tid == 0 ||
                   leaves[node.x-n_nodes].z - leaf_shifts[leaves[node.x-n_nodes].z] == tid ||
                   leaves[node.x-n_nodes].z - leaf_shifts[leaves[node.x-n_nodes].z-1] == tid);
            leaves[node.x-n_nodes].z = tid;
        }
        else {
            assert(tid <= nodes[node.x].z);
            assert(nodes[node.x].z - tid <= n_removed);
            assert(tid == 0 ||
                   nodes[node.x].z - leaf_shifts[nodes[node.x].z] == tid ||
                   nodes[node.x].z - leaf_shifts[nodes[node.x].z-1] == tid);
            nodes[node.x].z = tid;
        }

        if (node.y >= n_nodes) {
            assert(tid <= leaves[node.y-n_nodes].z);
            assert(leaves[node.y-n_nodes].z - tid <= n_removed);
            assert(tid == 0 ||
                   leaves[node.y-n_nodes].z - leaf_shifts[leaves[node.y-n_nodes].z] == tid ||
                   leaves[node.y-n_nodes].z - leaf_shifts[leaves[node.y-n_nodes].z-1] == tid);
            leaves[node.y-n_nodes].z = tid;
        }
        else {
            assert(tid <= nodes[node.y].z);
            assert(nodes[node.y].z - tid <= n_removed);
            assert(tid == 0 ||
                   nodes[node.y].z - leaf_shifts[nodes[node.y].z] == tid ||
                   nodes[node.y].z - leaf_shifts[nodes[node.y].z-1] == tid);
            nodes[node.y].z = tid;
        }

        tid += blockDim.x * gridDim.x;
    }
}

template <typename Float, typename Float4>
__global__ void find_AABBs_kernel(const int4* nodes,
                                  volatile Box* node_AABBs,
                                  const int4* leaves,
                                  volatile Box* leaf_AABBs,
                                  const size_t n_leaves,
                                  const Float4* spheres,
                                  unsigned int* g_flags)
{
    int tid, index, flag_index, block_lower, block_upper;
    int4 node;
    Float4 sphere;
    Float x_min, y_min, z_min;
    Float x_max, y_max, z_max;
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
            // nodes and leaves are both saved as int4's.
            // .x: first sphere index; .y: sphere count; .z: parent index.
            node = leaves[tid];
            sphere = spheres[node.x];

            // sphere.w == radius.
            x_max = sphere.x + sphere.w;
            y_max = sphere.y + sphere.w;
            z_max = sphere.z + sphere.w;

            x_min = sphere.x - sphere.w;
            y_min = sphere.y - sphere.w;
            z_min = sphere.z - sphere.w;

            for (int i=1; i<node.y; i++) {
                sphere = spheres[node.x+i];

                x_max = max(x_max, sphere.x + sphere.w);
                y_max = max(y_max, sphere.y + sphere.w);
                z_max = max(z_max, sphere.z + sphere.w);

                x_min = min(x_min, sphere.x - sphere.w);
                y_min = min(y_min, sphere.y - sphere.w);
                z_min = min(z_min, sphere.z - sphere.w);
            }

            leaf_AABBs[tid].tx = x_max;
            leaf_AABBs[tid].ty = y_max;
            leaf_AABBs[tid].tz = z_max;

            leaf_AABBs[tid].bx = x_min;
            leaf_AABBs[tid].by = y_min;
            leaf_AABBs[tid].bz = z_min;

            // Travel up the tree.  The second thread to reach a node writes
            // its AABB based on those of its children.  The first exits the
            // loop.
            index = node.z;
            // .x/.y: left/right child index; .z: parent index; .w: end index.
            node = nodes[index];
            in_block = (min(index, index + node.w) >= block_lower &&
                        max(index, index + node.w) <= block_upper);

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
                if (node.x >= n_leaves-1)
                    left_AABB = &leaf_AABBs[(node.x - (n_leaves-1))];
                else
                    left_AABB = &node_AABBs[node.x];

                if (node.y >= n_leaves-1)
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
                in_block = (min(index, index + node.w) >= block_lower &&
                            max(index, index + node.w) <= block_upper);

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
                 const thrust::device_vector<UInteger>& d_keys,
                 const int max_per_leaf=1)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.
    // TODO: Error if max_per_leaf >= n_keys

    size_t n_keys = d_keys.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_keys + BUILD_THREADS_PER_BLOCK-1)
                                        / BUILD_THREADS_PER_BLOCK));

    gpu::build_nodes_kernel<<<blocks,BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.level.data()),
        thrust::raw_pointer_cast(d_leaves.indices.data()),
        max_per_leaf,
        thrust::raw_pointer_cast(d_keys.data()),
        n_keys);
}

void compact_nodes(Nodes& d_nodes,
                   Leaves& d_leaves)
{
    thrust::device_vector<unsigned int> d_leaf_shifts(d_leaves.indices.size());
    thrust::transform_inclusive_scan(d_leaves.indices.begin(),
                                     d_leaves.indices.end(),
                                     d_leaf_shifts.begin(),
                                     flag_null_node(),
                                     thrust::plus<unsigned int>());
    const unsigned int N_removed = d_leaf_shifts.back();
    const size_t N_nodes = d_leaves.indices.size() - N_removed - 1;
    // Also try remove(_copy)_if with un-scanned flags as a stencil.
    // Then assert *(d_leaf_shifts.back()) == d_leaves.indices.size()
    gpu::copy_valid_nodes(d_nodes.hierarchy, N_nodes);
    gpu::copy_valid_nodes(d_leaves.indices, N_nodes+1);
    gpu::copy_valid_levels(d_nodes.level, N_nodes);
    d_nodes.AABB.resize(d_nodes.hierarchy.size());
    d_leaves.AABB.resize(d_leaves.indices.size());

    int blocks = min(MAX_BLOCKS, (int) ((N_nodes + SHIFTS_THREADS_PER_BLOCK-1)
                                        / SHIFTS_THREADS_PER_BLOCK));

    gpu::shift_tree_indices<<<blocks,SHIFTS_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_leaves.indices.data()),
        thrust::raw_pointer_cast(d_leaf_shifts.data()),
        N_removed,
        N_nodes);
}

template <typename Float4>
void find_AABBs(Nodes& d_nodes,
                Leaves& d_leaves,
                const thrust::device_vector<Float4>& d_spheres)
{
    size_t n_leaves = d_leaves.indices.size();

    thrust::device_vector<unsigned int> d_AABB_flags(n_leaves-1);

    int blocks = min(MAX_BLOCKS, (int) ((n_leaves + AABB_THREADS_PER_BLOCK-1)
                                        / AABB_THREADS_PER_BLOCK));

    gpu::find_AABBs_kernel<float, Float4><<<blocks,AABB_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        thrust::raw_pointer_cast(d_leaves.indices.data()),
        thrust::raw_pointer_cast(d_leaves.AABB.data()),
        n_leaves,
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_AABB_flags.data())
    );
}


} // namespace grace
