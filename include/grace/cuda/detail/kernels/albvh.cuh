#pragma once

#include "grace/cuda/detail/device/loadstore.cuh"

#include "grace/cuda/detail/kernel_config.h"

#include "grace/cuda/bvh.cuh"

#include "grace/aabb.h"
#include "grace/config.h"
#include "grace/error.h"
#include "grace/vector.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/swap.h>

// CUDA math constants.
#include <math_constants.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace grace {

namespace detail {

//
// ALBVH helper functions
//

GRACE_DEVICE bool albvh_is_left_child(int2 node, int parent_index) {
    return node.y == parent_index;
}

GRACE_DEVICE CudaBvhNode albvh_decode_node(CudaBvhNode node) {
    CudaBvhNode decoded(node);
    decoded.set_left_child(-1 * node.left_child() - 1);
    decoded.set_right_child(-1 * node.right_child() - 1);

    return decoded;
}

GRACE_DEVICE CudaBvhNode albvh_encode_node(CudaBvhNode node) {
    CudaBvhNode encoded(node);
    encoded.set_left_child(-1 * (node.left_child() + 1));
    encoded.set_right_child(-1 * (node.right_child() + 1));
    return encoded;
}

// Return the parent index in terms of the base nodes, given the actual parent
// index and the base nodes covered by the current node.
GRACE_DEVICE int albvh_logical_parent(CudaBvhNode node, int2 g_node, int g_parent_index)
{
    return albvh_is_left_child(g_node, g_parent_index) ? node.last_leaf() : node.first_leaf() - 1;
}

// Find the parent index of a node.
// node contains the left- and right-most primitives it spans.
// deltas is an array of per-primitive delta values.
template <typename DeltaIter, typename DeltaComp>
GRACE_DEVICE int albvh_node_parent(int2 node, DeltaIter deltas, DeltaComp comp)
{
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;
    DeltaType deltaL = deltas[node.x - 1];
    DeltaType deltaR = deltas[node.y];

    int parent_index;
    if (comp(deltaL, deltaR)) {
        parent_index = node.x - 1;
    }
    else {
        parent_index = node.y;
    }

    return parent_index;
}

// Find the parent index of a node.
// node contains the left- and right-most primitives it spans.
// deltas is an array of per-leaf delta values.
template <typename DeltaIter, typename DeltaComp>
GRACE_DEVICE int albvh_node_parent(CudaBvhNode node, DeltaIter deltas, DeltaComp comp)
{
    return albvh_node_parent(make_int2(node.first_leaf(), node.last_leaf()), deltas, comp);
}

GRACE_DEVICE bool out_of_block(int index, int low, int high) {
    return (index < low || index >= high);
}

//-----------------------------------------------------------------------------
// CUDA ALBVH kernels.
//-----------------------------------------------------------------------------

template <typename KeyIter, typename DeltaIter, typename DeltaFunc>
__global__ void compute_deltas_kernel(
    KeyIter keys,
    const size_t n_keys,
    DeltaIter deltas,
    const DeltaFunc delta_func)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid <= n_keys)
    {
        // The range [-1, n_keys) is valid for querying node_delta.
        deltas[tid] = delta_func(tid - 1, keys, n_keys);
        tid += blockDim.x * gridDim.x;
    }
}

// Two template parameters as DeltaIter _may_ be const_iterator or const T*.
template<typename DeltaIter, typename LeafDeltaIter>
__global__ void copy_leaf_deltas_kernel(
    const CudaBvhLeaf* leaves,
    const size_t n_leaves,
    DeltaIter all_deltas,
    LeafDeltaIter leaf_deltas)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // The range [-1, n_leaves) is valid for querying node_delta.
    // leaf/all_deltas[0] corresponds to the -1 case.
    if (tid == 0)
        leaf_deltas[0] = all_deltas[0];
    ++all_deltas;
    ++leaf_deltas;

    while (tid < n_leaves)
    {
        CudaBvhLeaf leaf = leaves[tid];
        int last_idx = leaf.last_primitive();
        leaf_deltas[tid] = all_deltas[last_idx];

        tid += blockDim.x * gridDim.x;
    }
}

template <typename DeltaIter, typename DeltaComp>
__global__ void build_leaves_kernel(
    int2* nodes,
    const size_t n_nodes,
    DeltaIter deltas,
    const int max_per_leaf,
    const DeltaComp delta_comp)
{
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;

    extern __shared__ int flags[];

    const size_t n_leaves = n_nodes + 1;

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    ++deltas;

    int bid = blockIdx.x;
    int tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;
    // while() and an inner for() to ensure all threads in a block hit the
    // __syncthreads() and wipe the flags.
    while (bid * grace::BUILD_THREADS_PER_BLOCK < n_leaves)
    {
        // Zero all SMEM flags at start of first loop and at end of subsequent
        // loops.
        __syncthreads();
        for (int i = threadIdx.x;
             i < grace::BUILD_THREADS_PER_BLOCK + max_per_leaf;
             i += grace::BUILD_THREADS_PER_BLOCK)
        {
            flags[i] = 0;
        }
        __syncthreads();

        // [low, high) leaf indices covered by this block, including the
        // max_per_leaf buffer.
        int low = bid * grace::BUILD_THREADS_PER_BLOCK;
        int high = min((bid + 1) * grace::BUILD_THREADS_PER_BLOCK + max_per_leaf,
                       (int)n_leaves);

        for (int idx = tid; idx < high; idx += grace::BUILD_THREADS_PER_BLOCK)
        {
            int cur_index = idx;
            int parent_index;

            // Compute the current leaf's parent index and write associated data
            // to the parent. The leaf is not actually written.
            int left = cur_index;
            int right = cur_index;
            DeltaType delta_L = deltas[left - 1];
            DeltaType delta_R = deltas[right];
            int* addr;
            int end;
            if (delta_comp(delta_L, delta_R))
            {
                // Leftward node is parent.
                parent_index = left - 1;
                // Current leaf is a right child.
                addr = &(nodes[parent_index].y);
                end = right;
            }
            else {
                // Rightward node is parent.
                parent_index = right;
                // Current leaf is a left child.
                addr = &(nodes[parent_index].x);
                end = left;
            }

            // If the leaf's parent is outside this block do not write anything;
            // an adjacent block will follow this path.
            if (parent_index < low || parent_index >= high)
                continue;

            // Normal store.  Other threads in this block can read from L1 if
            // they get a hit.  No requirement for global coherency.
            *addr = end;

            // Travel up the tree.  The second thread to reach a node writes its
            // left or right end to its parent.  The first exits the loop.
            __threadfence_block();
            GRACE_ASSERT(parent_index - low >= 0);
            GRACE_ASSERT(parent_index - low < grace::BUILD_THREADS_PER_BLOCK + max_per_leaf);
            unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
            GRACE_ASSERT(flag < 2);
            bool first_arrival = (flag == 0);

            while (!first_arrival)
            {
                cur_index = parent_index;

                // We are certain that a thread in this block has already
                // written the other child of the current node, so we can read
                // from L1 cache if we get a hit.
                int2 left_right = nodes[parent_index];
                int left = left_right.x;
                int right = left_right.y;

                // Only the left-most leaf can have an index of 0, and only the
                // right-most leaf can have an index of n_leaves - 1.
                GRACE_ASSERT(left >= 0);
                GRACE_ASSERT(right > 0);
                GRACE_ASSERT(left < n_leaves - 1);
                GRACE_ASSERT(right < n_leaves);

                int size = right - left + 1;
                if (size > max_per_leaf) {
                    // Both children of the current node must be leaves.
                    // Stop traveling up the tree and continue with outer loop.
                    break;
                }

                DeltaType delta_L = deltas[left - 1];
                DeltaType delta_R = deltas[right];
                int* addr;
                int end;
                // Compute the current node's parent index and write associated
                // data.
                if (delta_comp(delta_L, delta_R))
                {
                    // Leftward node is parent.
                    parent_index = left - 1;
                    // Current node is a right child.
                    addr = &(nodes[parent_index].y);
                    end = right;
                }
                else {
                    // Rightward node is parent.
                    parent_index = right;
                    // Current node is a left child.
                    addr = &(nodes[parent_index].x);
                    end = left;

                }

                if (parent_index < low || parent_index >= high) {
                    // Parent node outside this block's boundaries, exit without
                    // writing the unreachable node and flag.
                    break;
                }

                // Normal store.
                *addr = end;

                __threadfence_block();

                unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
                first_arrival = (flag == 0);

                GRACE_ASSERT(parent_index - low >= 0);
                GRACE_ASSERT(parent_index - low < grace::BUILD_THREADS_PER_BLOCK + max_per_leaf);
                GRACE_ASSERT(flag < 2);
            } // while (!first_arrival)
        } // for idx = [threadIdx.x, BUILD_THREADS_PER_BLOCK + max_per_leaf)
        bid += gridDim.x;
        tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;
    } // while (bid * BUILD_THREADS_PER_BLOCK < n_leaves)
    return;
}

__global__ void write_leaves_kernel(
    const int2* nodes,
    const size_t n_nodes,
    CudaBvhLeaf* big_leaves,
    const int max_per_leaf)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n_nodes)
    {
        int2 node = nodes[tid];
        int left = node.x;
        int right = node.y;

        // If left or right are 0, size may be incorrect.
        int size = right - left + 1;

        // If left is 0, left_size may be incorrect:
        // we cannot differentiate between an unwritten node.x and one
        // written as 0.
        int left_size = tid - left + 1;
        // This is guaranteed to be sufficiently correct:
        // right == 0 means node.y was unwritten, and the right child is
        // therefore not a leaf, so its size is set accordingly.
        int right_size = (right > 0 ? right - tid : max_per_leaf + 1);

        // *These are both guarranteed to be correct*:
        // If left_size was computed incorrectly, then the true value of
        // node.x is not zero, and thus node.x was unwritten.  This requires
        // that the left child not be a leaf, and hence the node index (tid)
        // must be > max_per_leaf, resulting in left_leaf = false.
        //
        // right_leaf follows from the correctness of right_size.
        bool left_leaf = (left_size <= max_per_leaf);
        bool right_leaf = (right_size <= max_per_leaf);

        // If only one child is to be written, we are certain it should be,
        // as the current node's (unknown) size must be > max_per_leaf.
        // Otherwise, we write only if the current node cannot be a leaf.
        bool write_check = left_leaf != right_leaf ? true :
                                                    (size > max_per_leaf);

        // NOTE: size is guaranteed accurate only if both left_leaf and
        // right_leaf are true, but if they are both false no write occurs
        // anyway because of the && below.
        CudaBvhLeaf leaf;
        if (left_leaf && write_check) {
            leaf.set_first_primitive(left);
            leaf.set_size(left_size);
            big_leaves[left] = leaf;
        }
        if (right_leaf && write_check) {
            leaf.set_first_primitive(tid + 1);
            leaf.set_size(right_size);
            big_leaves[right] = leaf;
        }

        tid += blockDim.x * gridDim.x;
    }
}

template <
    typename PrimitiveIter,
    typename DeltaIter,
    typename DeltaComp,
    typename AABBFunc
    >
__global__ void build_nodes_slice_kernel(
    CudaBvhNode* nodes,
    const size_t n_nodes,
    const CudaBvhLeaf* leaves,
    const size_t n_leaves,
    PrimitiveIter primitives,
    const int* base_indices,
    const size_t n_base_nodes,
    int* root_index,
    DeltaIter deltas,
    const int max_per_node,
    int* new_base_indices,
    const DeltaComp delta_comp,
    const AABBFunc aabb_op)
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type PrimitiveType;
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;

    extern __shared__ int SMEM[];

    int* flags = SMEM;
    volatile int2* sm_nodes
        = (int2*)&flags[grace::BUILD_THREADS_PER_BLOCK + max_per_node];

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    ++deltas;

    int bid = blockIdx.x;
    int tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;

    // while() and inner for() loops to ensure all threads in a block hit the
    // __syncthreads().
    while (bid * grace::BUILD_THREADS_PER_BLOCK < n_base_nodes)
    {
        // Zero all SMEM flags and node data at start of first loop and at end
        // of subsequent loops.
        __syncthreads();
        for (int i = threadIdx.x;
             i < grace::BUILD_THREADS_PER_BLOCK + max_per_node;
             i += grace::BUILD_THREADS_PER_BLOCK)
        {
            flags[i] = 0;
            // 'error: no operator "=" matches ... volatile int2 = int2':
            // sm_nodes[i] = make_int2(0, 0);
            sm_nodes[i].x = 0;
            sm_nodes[i].y = 0;
        }
        __syncthreads();

        // The compressed/logical node indices for this block cover the range
        // [low, high), including the max_per_node buffer.
        int low = bid * grace::BUILD_THREADS_PER_BLOCK;
        int high = min((bid + 1) * grace::BUILD_THREADS_PER_BLOCK + max_per_node,
                       (int)n_base_nodes);

        for (int idx = tid; idx < high; idx += grace::BUILD_THREADS_PER_BLOCK)
        {
            // For the tree climb, we start at base nodes, treating them as
            // leaves.  The left/right values refer to the left- and right-most
            // *base nodes* covered by the current node.  (g_left and g_right
            // contain the left- and right-most *actual* leaf indices.)
            // The current index (i.e. the idx-th base node) is thus a logical,
            // or compressed node index, and is used for writing to shared
            // memory.
            int cur_index = idx;
            int parent_index;

            // These are real node indices (for writing to global memory).
            int g_cur_index = base_indices[cur_index];
            int g_parent_index;

            // Node index can be >= n_nodes if a leaf.
            GRACE_ASSERT(g_cur_index < n_nodes + n_leaves);

            int g_left, g_right;
            AABBf node_aabb;
            if (g_cur_index < n_nodes) {
                CudaBvhNode node = nodes[g_cur_index];
                node_aabb = node.AABB();

                g_left = node.first_leaf();
                g_right = node.last_leaf();
            }
            else {
                CudaBvhLeaf leaf = leaves[g_cur_index - n_nodes];

                for (int i = 0; i < leaf.size(); ++i) {
                    PrimitiveType prim = primitives[leaf.first_primitive() + i];

                    AABBf prim_aabb;
                    aabb_op(prim, &prim_aabb);
                    node_aabb = aabb_union(node_aabb, prim_aabb);
                }

                // Recall that leaf left/right limits are equal to the leaf
                // index, minus the leaf-identifying offset.
                g_left = g_cur_index - n_nodes;
                g_right = g_cur_index - n_nodes;
            }

            // Note, they should never be equal.
            GRACE_ASSERT(node_aabb.min.x < node_aabb.max.x);
            GRACE_ASSERT(node_aabb.min.y < node_aabb.max.y);
            GRACE_ASSERT(node_aabb.min.z < node_aabb.max.z);

            // Only the left-most leaf can have an index of 0, and only the
            // right-most leaf can have an index of n_leaves - 1.
            if (g_cur_index < n_nodes) {
                GRACE_ASSERT(g_left >= 0);
                GRACE_ASSERT(g_right > 0);
                GRACE_ASSERT(g_left < n_leaves - 1);
                GRACE_ASSERT(g_right < n_leaves);
            }
            else {
                GRACE_ASSERT(g_left >= 0);
                GRACE_ASSERT(g_right >= 0);
                GRACE_ASSERT(g_left < n_leaves);
                GRACE_ASSERT(g_right < n_leaves);
            }

            int left = cur_index;
            int right = cur_index;

            DeltaType delta_L = deltas[g_left - 1];
            DeltaType delta_R = deltas[g_right];
            // Compute the current node's parent index and write-to locations.
            if (delta_comp(delta_L, delta_R))
            {
                // Leftward node is parent.
                // Current node is a right child.
                g_parent_index = g_left - 1;
                parent_index = left - 1;
            }
            else
            {
                // Rightward node is parent.
                // Current node is a left child.
                g_parent_index = g_right;
                parent_index = right;
            }

            // If the leaf's parent is outside this block do not write anything;
            // an adjacent block will follow this path.
            if (out_of_block(parent_index, low, high))
                continue;

            // Parent index must be at a valid location for writing to sm_nodes
            // and the SMEM flags.
            GRACE_ASSERT(parent_index - low >= 0);
            GRACE_ASSERT(parent_index - low < grace::BUILD_THREADS_PER_BLOCK + max_per_node);

            if (delta_comp(delta_L, delta_R))
            {
                nodes[g_parent_index].set_right_child(g_cur_index);
                // The -ve end index encodes the fact that the node was written
                // this iteration.
                nodes[g_parent_index].set_last_leaf(-1 * (right + 1));
                nodes[g_parent_index].set_right_AABB(node_aabb);
                // Right-most global node index.
                sm_nodes[parent_index - low].y = g_right;
            }
            else
            {
                nodes[g_parent_index].set_left_child(g_cur_index);
                // The -ve end index encodes the fact that the node was written
                // this iteration.
                nodes[g_parent_index].set_first_leaf(-1 * (left + 1));
                nodes[g_parent_index].set_left_AABB(node_aabb);
                // Left-most global node index.
                sm_nodes[parent_index - low].x = g_left;
            }

            // Travel up the tree.  The second thread to reach a node writes its
            // logical/compressed left or right end to its parent; the first
            // exits the loop.
            __threadfence_block();
            unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
            GRACE_ASSERT(flag < 2);
            bool first_arrival = (flag == 0);

            while (!first_arrival)
            {
                cur_index = parent_index;
                g_cur_index = g_parent_index;

                GRACE_ASSERT(cur_index < n_base_nodes);
                GRACE_ASSERT(g_cur_index < n_nodes);

                // We are certain that a thread in this block has already
                // *written* the other child of the current node, so we can read
                // from L1 if we get a cache hit.
                CudaBvhNode node = nodes[g_cur_index];

                // 'error: class "int2" has no suitable copy constructor'
                // int2 left_right = sm_nodes[cur_index - low];
                // int g_left = left_right.x;
                // int g_right = left_right.y;
                int g_left = sm_nodes[cur_index - low].x;
                int g_right = sm_nodes[cur_index - low].y;
                GRACE_ASSERT(g_left >= 0);
                GRACE_ASSERT(g_right > 0);
                GRACE_ASSERT(g_left < n_leaves - 1);
                GRACE_ASSERT(g_right < n_leaves);

                // Undo the -ve sign encoding.
                int left = -1 * node.first_leaf() - 1;
                int right = -1 * node.last_leaf() - 1;
                // We are the second thread in this block to reach this node.
                // Both of its logical/compressed end-indices must also be in
                // this block.
                GRACE_ASSERT(left >= low);
                GRACE_ASSERT(right > low);
                GRACE_ASSERT(left < high - 1);
                GRACE_ASSERT(right < high);

                // Even if this is true, the following compacted/logical size
                // test can be false.
                if (g_right - g_left == n_leaves - 1)
                    *root_index = g_cur_index;

                // Check our compacted/logical size, and exit the loop if we are
                // large enough to become a base layer in the next iteration.
                if (right - left + 1 > max_per_node) {
                    // Both children of the current node must be leaves.  Stop
                    // travelling up the tree and continue with the outer loop.
                    break;
                }

                // Again, L1 data will be accurate.
                AABBf node_aabb = node.AABB();

                GRACE_ASSERT(node_aabb.min.x < node_aabb.max.x);
                GRACE_ASSERT(node_aabb.min.y < node_aabb.max.y);
                GRACE_ASSERT(node_aabb.min.z < node_aabb.max.z);

                DeltaType delta_L = deltas[g_left - 1];
                DeltaType delta_R = deltas[g_right];
                if (delta_comp(delta_L, delta_R))
                {
                    // Leftward node is parent.
                    // Current node is a right child.
                    parent_index = left - 1;
                    g_parent_index = g_left - 1;

                }
                else {
                    // Rightward node is parent.
                    // Current node is a left child.
                    parent_index = right;
                    g_parent_index = g_right;
                }

                if (out_of_block(parent_index, low, high)) {
                    // Parent node outside this block's boundaries.  Either a
                    // thread in an adjacent block will follow this path, or the
                    // current node will be a base node in the next iteration.
                    break;
                }

                if (delta_comp(delta_L, delta_R))
                {
                    nodes[g_parent_index].set_right_child(g_cur_index);
                    // The -ve end index encodes the fact that the node was written
                    // this iteration.
                    nodes[g_parent_index].set_last_leaf(-1 * (right + 1));
                    nodes[g_parent_index].set_right_AABB(node_aabb);
                    // Right-most global node index.
                    sm_nodes[parent_index - low].y = g_right;
                }
                else
                {
                    nodes[g_parent_index].set_left_child(g_cur_index);
                    // The -ve end index encodes the fact that the node was written
                    // this iteration.
                    nodes[g_parent_index].set_first_leaf(-1 * (left + 1));
                    nodes[g_parent_index].set_left_AABB(node_aabb);
                    // Left-most global node index.
                    sm_nodes[parent_index - low].x = g_left;
                }

                __threadfence_block();
                GRACE_ASSERT(parent_index - low >= 0);
                GRACE_ASSERT(parent_index - low < grace::BUILD_THREADS_PER_BLOCK + max_per_node);
                unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
                GRACE_ASSERT(flag < 2);
                first_arrival = (flag == 0);
            } // while (!first_arrival)
        } // for idx = [threadIdx.x, BUILD_THREADS_PER_BLOCK + max_per_node)

        bid += gridDim.x;
        tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;
    } // while (bid * BUILD_THREADS_PER_BLOCK < n_base_nodes)
    return;
}

__global__ void fill_output_queue(
    const CudaBvhNode* nodes,
    const size_t n_nodes,
    const int max_per_node,
    int* new_base_indices)
{
    int tid = threadIdx.x + blockIdx.x * grace::BUILD_THREADS_PER_BLOCK;

    while (tid < n_nodes)
    {
        CudaBvhNode node = nodes[tid];

        // The left/right values are negative IFF they were written in this
        // iteration.  A negative index represents a logical/compacted index.
        bool left_written = (node.first_leaf() < 0);
        bool right_written = (node.last_leaf() < 0);

        // Undo the -ve encoding.
        int left = -1 * node.first_leaf() - 1;
        int right = -1 * node.last_leaf() - 1;

        if (left_written != right_written) {
            // Only one child was written; it must be placed in the output queue
            // at a *unique* location.
            int index = left_written ? left : right;
            GRACE_ASSERT(new_base_indices[index] == -1);
            new_base_indices[index] = left_written ? node.left_child() : node.right_child();
        }
        else if (left_written) {
            // Both were written, so the current node must be added to the
            // output queue IFF it is sufficiently large; that is, if its size
            // is such that it would have caused the thread to end its tree
            // climb.
            // Again, the location we write to must be unique.
            int size = right - left + 1;
            if (size > max_per_node) {
                GRACE_ASSERT(new_base_indices[left] == -1);
                new_base_indices[left] = tid;
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void fix_node_ranges(
    CudaBvhNode* nodes,
    const size_t n_nodes,
    const CudaBvhLeaf* leaves,
    const int* old_base_indices)
{
    int tid = threadIdx.x + blockIdx.x * grace::BUILD_THREADS_PER_BLOCK;

    while (tid < n_nodes)
    {
        CudaBvhNode node = nodes[tid];

        // We only need to fix nodes whose ranges were written, *in full*, this
        // iteration.
        int first = node.first_leaf();
        int last = node.last_leaf();
        if (first < 0 && last < 0)
        {
            int left = first < 0 ? (-1 * first - 1) : first;
            int right = last < 0 ? (-1 * last -  1) : last;

            // All base nodes have correct range indices.  If we know our
            // left/right-most base node, we can therefore find our
            // left/right-most leaf indices.
            // Note that for leaves, the left and right ranges are simply the
            // (corrected) index of the leaf.
            int index = old_base_indices[left];
            left = index < n_nodes ? nodes[index].first_leaf() : index - n_nodes;

            index = old_base_indices[right];
            right = index < n_nodes ? nodes[index].last_leaf() : index - n_nodes;

            // Only the left-most leaf can have index 0.
            GRACE_ASSERT(left >= 0);
            GRACE_ASSERT(left < n_nodes);
            GRACE_ASSERT(right > 0);
            // Only the right-most leaf can have index n_leaves - 1 == n_nodes.
            GRACE_ASSERT(right <= n_nodes);

            node.set_first_leaf(left);
            node.set_last_leaf(right);
            nodes[tid] = node;
        }

        tid += blockDim.x * gridDim.x;
    }
}

//-----------------------------------------------------------------------------
// C-like wrappers for ALBVH kernels.
//-----------------------------------------------------------------------------

// Two template parameters as DeltaIter _may_ be const_iterator or const T*.
template<typename DeltaIter, typename LeafDeltaIter>
GRACE_HOST void copy_leaf_deltas(
    const thrust::device_vector<CudaBvhLeaf>& d_leaves,
    DeltaIter d_all_deltas_iter,
    LeafDeltaIter d_leaf_deltas_iter)
{
    const int blocks = std::min(grace::MAX_BLOCKS,
                                (int)((d_leaves.size() + 511) / 512 ));
    copy_leaf_deltas_kernel<<<blocks, 512>>>(
        thrust::raw_pointer_cast(d_leaves.data()),
        d_leaves.size(),
        d_all_deltas_iter,
        d_leaf_deltas_iter);
    GRACE_KERNEL_CHECK();
}

template <typename DeltaIter, typename DeltaComp>
GRACE_HOST void build_leaves(
    thrust::device_vector<int2>& d_tmp_nodes,
    thrust::device_vector<CudaBvhLeaf>& d_tmp_leaves,
    const int max_per_leaf,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp)
{
    const size_t n_leaves = d_tmp_leaves.size();
    const size_t n_nodes = n_leaves - 1;

    if (n_leaves <= max_per_leaf) {
        const std::string msg
            = "max_per_leaf must be less than the total number of primitives.";
        throw std::invalid_argument(msg);
    }

    int blocks = std::min(grace::MAX_BLOCKS,
                          (int)((n_leaves + grace::BUILD_THREADS_PER_BLOCK - 1)
                                 / grace::BUILD_THREADS_PER_BLOCK));
    int smem_size = sizeof(int) * (grace::BUILD_THREADS_PER_BLOCK + max_per_leaf);

    build_leaves_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK, smem_size>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        d_deltas_iter,
        max_per_leaf,
        delta_comp);
    GRACE_KERNEL_CHECK();

    blocks = std::min(grace::MAX_BLOCKS,
                      (int)((n_nodes + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));

    write_leaves_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_tmp_leaves.data()),
        max_per_leaf);
    GRACE_KERNEL_CHECK();
}

GRACE_HOST void remove_empty_leaves(thrust::device_vector<CudaBvhLeaf>& d_leaves)
{
    // A transform_reduce (with unary op 'is_valid_node()') followed by a
    // copy_if (with predicate 'is_valid_node()') actually seems slightly faster
    // than the below.  However, this method does not require a temporary leaves
    // array, which would be the largest temporary memory allocation in the
    // build process.

    // Try
    // thrust::remove(.., .., CudaBvhLeaf());

    typedef typename thrust::device_vector<CudaBvhLeaf>::iterator LeafIter;
    LeafIter end = thrust::remove_if(d_leaves.begin(),
                                     d_leaves.end(),
                                     detail::is_empty_cuda_bvh_node());

    const size_t n_new_leaves = end - d_leaves.begin();
    d_leaves.resize(n_new_leaves);
}

template <
    typename PrimitiveIter,
    typename DeltaIter,
    typename DeltaComp,
    typename AABBFunc
    >
GRACE_HOST void build_nodes(
    CudaBvh& bvh,
    PrimitiveIter d_prims_iter,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp,
    const AABBFunc aabb_op)
{
    const size_t n_leaves = bvh.num_leaves();
    const size_t n_nodes = bvh.num_nodes();
    detail::Bvh_ref<CudaBvh> bvh_ref(bvh);

    thrust::device_vector<int> d_root_index(1);
    d_root_index[0] = -1;
    int* d_root_index_ptr = thrust::raw_pointer_cast(d_root_index.data());

    thrust::device_vector<int> d_queue1(n_leaves);
    thrust::device_vector<int> d_queue2(n_leaves);
    // The first input queue of base layer nodes is simply all the leaves.
    // The first leaf has index n_nodes.
    thrust::sequence(d_queue1.begin(), d_queue1.end(), n_nodes);

    typedef thrust::device_vector<int>::iterator QIter;
    QIter in_q_begin  = d_queue1.begin();
    QIter in_q_end    = d_queue1.end();
    QIter out_q_begin = d_queue2.begin();
    QIter out_q_end   = d_queue2.end();

    int* d_in_ptr = thrust::raw_pointer_cast(d_queue1.data());
    int* d_out_ptr = thrust::raw_pointer_cast(d_queue2.data());

    while (in_q_end - in_q_begin > 1)
    {
        const size_t n_in = in_q_end - in_q_begin;

        // Output queue is always filled with invalid values so we can remove them
        // and use it as an input in the next iteration.
        thrust::fill(out_q_begin, out_q_end, -1);

        int blocks = std::min(grace::MAX_BLOCKS,
                              (int)((n_in + grace::BUILD_THREADS_PER_BLOCK - 1)
                                     / grace::BUILD_THREADS_PER_BLOCK));
        // SMEM has to cover for BUILD_THREADS_PER_BLOCK + max_per_leaf flags
        // AND int2 nodes.
        int smem_size = (sizeof(int) + sizeof(int2))
                        * (grace::BUILD_THREADS_PER_BLOCK + bvh.max_per_leaf());
        build_nodes_slice_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK, smem_size>>>(
            thrust::raw_pointer_cast(bvh_ref.nodes().data()),
            n_nodes,
            thrust::raw_pointer_cast(bvh_ref.leaves().data()),
            n_leaves,
            d_prims_iter,
            d_in_ptr,
            n_in,
            d_root_index_ptr,
            d_deltas_iter,
            bvh.max_per_leaf(), // This can actually be anything.
            d_out_ptr,
            delta_comp,
            aabb_op);
        GRACE_KERNEL_CHECK();

        blocks = std::min(grace::MAX_BLOCKS,
                          (int)((n_nodes + grace::BUILD_THREADS_PER_BLOCK - 1)
                                 / grace::BUILD_THREADS_PER_BLOCK));
        fill_output_queue<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(bvh_ref.nodes().data()),
            n_nodes,
            bvh.max_per_leaf(),
            d_out_ptr);
        GRACE_KERNEL_CHECK();

        fix_node_ranges<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(bvh_ref.nodes().data()),
            n_nodes,
            thrust::raw_pointer_cast(bvh_ref.leaves().data()),
            d_in_ptr);
        GRACE_KERNEL_CHECK();

        QIter end = thrust::remove(out_q_begin, out_q_end, -1);
        out_q_end = end;

        // New output queue becomes the input queue next iteration.
        thrust::swap(d_in_ptr, d_out_ptr);
        thrust::swap(in_q_begin, out_q_begin);
        in_q_end = out_q_end;
        // The new output queue need be no larger than the input queue. This is
        // worth doing since we need to fill it with -1's.
        out_q_end = out_q_begin + (in_q_end - in_q_begin);
    }

    int root_index = d_root_index[0];
    GRACE_ASSERT(root_index != -1);

    bvh.set_root_index(root_index);
}

} // namespace detail


//-----------------------------------------------------------------------------
// User functions for ALBVH building.
//-----------------------------------------------------------------------------

template<typename KeyIter, typename DeltaIter, typename DeltaFunc>
GRACE_HOST void compute_deltas(
    KeyIter d_keys_iter,
    const size_t N_keys,
    DeltaIter d_deltas_iter,
    const DeltaFunc delta_func)
{
    const size_t N_deltas = N_keys + 1;
    int blocks = std::min(grace::MAX_BLOCKS, (int)((N_deltas + 512 - 1) / 512));
    detail::compute_deltas_kernel<<<blocks, 512>>>(
        d_keys_iter,
        N_keys,
        d_deltas_iter,
        delta_func);
    GRACE_KERNEL_CHECK();
}

template<typename KeyType, typename DeltaType, typename DeltaFunc>
GRACE_HOST void compute_deltas(
    const thrust::device_vector<KeyType>& d_keys,
    thrust::device_vector<DeltaType>& d_deltas,
    const DeltaFunc delta_func)
{
    GRACE_ASSERT(d_keys.size() + 1 == d_deltas.size());

    const KeyType* keys_ptr = thrust::raw_pointer_cast(d_keys.data());
    DeltaType* deltas_ptr = thrust::raw_pointer_cast(d_deltas.data());

    compute_deltas(keys_ptr, d_keys.size(), deltas_ptr, delta_func);
}

template <
    typename PrimitiveIter,
    typename DeltaIter,
    typename DeltaComp,
    typename AABBFunc
    >
GRACE_HOST void build_ALBVH(
    CudaBvh& bvh,
    PrimitiveIter d_prims_iter,
    const size_t num_primitives,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp,
    const AABBFunc aabb_op,
    const bool wipe = false)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.

    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;

    const size_t n_leaves = num_primitives;
    const size_t n_nodes = n_leaves - 1;

    detail::Bvh_ref<CudaBvh> bvh_ref(bvh);

    // In case this ever changes.
    GRACE_ASSERT(sizeof(int4) == sizeof(float4));

    if (wipe) {
        thrust::fill(bvh_ref.nodes().begin(), bvh_ref.nodes().end(),
                     detail::CudaBvhNode());
        thrust::fill(bvh_ref.leaves().begin(), bvh_ref.leaves().end(),
                     detail::CudaBvhLeaf());
    }

    thrust::device_vector<detail::CudaBvhLeaf> d_tmp_leaves(n_leaves);
    thrust::device_vector<int2> d_tmp_nodes(n_nodes);

    detail::build_leaves(d_tmp_nodes, d_tmp_leaves, bvh.max_per_leaf(),
                         d_deltas_iter, delta_comp);
    detail::remove_empty_leaves(d_tmp_leaves);

    bvh_ref.leaves() = d_tmp_leaves;

    const size_t n_new_leaves = bvh.num_leaves();
    const size_t n_new_nodes = n_new_leaves - 1;
    bvh_ref.nodes().resize(n_new_nodes);

    thrust::device_vector<DeltaType> d_new_deltas(n_new_leaves + 1);
    DeltaType* new_deltas_ptr = thrust::raw_pointer_cast(d_new_deltas.data());

    detail::copy_leaf_deltas(bvh_ref.leaves(), d_deltas_iter, new_deltas_ptr);
    detail::build_nodes(bvh, d_prims_iter, new_deltas_ptr, delta_comp,
                        aabb_op);
}

template <
    typename PrimitiveType,
    typename DeltaType,
    typename DeltaComp,
    typename AABBFunc
    >
GRACE_HOST void build_ALBVH(
    CudaBvh& bvh,
    const thrust::device_vector<PrimitiveType>& d_primitives,
    const thrust::device_vector<DeltaType>& d_deltas,
    const DeltaComp delta_comp,
    const AABBFunc aabb_op,
    const bool wipe = false)
{
    const PrimitiveType* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    const DeltaType* deltas_ptr = thrust::raw_pointer_cast(d_deltas.data());

    build_ALBVH(bvh, prims_ptr, d_primitives.size(), deltas_ptr,
                delta_comp, aabb_op, wipe);
}

// Specialized with DeltaComp = thrust::less<DeltaType>
template <typename PrimitiveIter, typename DeltaIter, typename AABBFunc>
GRACE_HOST void build_ALBVH(
    CudaBvh& bvh,
    PrimitiveIter d_prims_iter,
    const size_t num_primitives,
    DeltaIter d_deltas_iter,
    const AABBFunc aabb_op,
    const bool wipe = false)
{
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;
    typedef typename thrust::less<DeltaType> DeltaComp;

    build_ALBVH(bvh, d_prims_iter, num_primitives, d_deltas_iter,
                DeltaComp(), aabb_op, wipe);
}

// Specialized with DeltaComp = thrust::less<DeltaType>
template <typename PrimitiveType, typename DeltaType, typename AABBFunc>
GRACE_HOST void build_ALBVH(
    CudaBvh& bvh,
    const thrust::device_vector<PrimitiveType>& d_primitives,
    const thrust::device_vector<DeltaType>& d_deltas,
    const AABBFunc aabb_op,
    const bool wipe = false)
{
    typedef typename thrust::less<DeltaType> DeltaComp;
    const PrimitiveType* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    const DeltaType* deltas_ptr = thrust::raw_pointer_cast(d_deltas.data());

    build_ALBVH(bvh, prims_ptr, d_primitives.size(), deltas_ptr,
                DeltaComp(), aabb_op, wipe);
}

} // namespace grace
