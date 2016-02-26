#pragma once

// CUDA math constants.
#include <math_constants.h>

#include <stdexcept>
#include <string>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/swap.h>

#include "../device/loadstore.cuh"
#include "../error.h"
#include "../kernel_config.h"
#include "../nodes.h"
#include "../types.h"

namespace grace {

namespace ALBVH {

// Compute the union of the two AABBs packed into a node (as three float4s).
GRACE_DEVICE void AABB_union(float4 AABB1, float4 AABB2, float4 AABB3,
                             float3 *bottom, float3 *top)
{
    bottom->x = min(AABB1.x, AABB2.x);
    top->x    = max(AABB1.y, AABB2.y);

    bottom->y = min(AABB1.z, AABB2.z);
    top->y    = max(AABB1.w, AABB2.w);

    bottom->z = min(AABB3.x, AABB3.z);
    top->z    = max(AABB3.y, AABB3.w);
}

// Compute the union of the two AABBs packed into a node (as three float4s).
GRACE_DEVICE void AABB_union(float4 AABB1, float4 AABB2, float4 AABB3,
                             float2 *AABBx, float2 *AABBy, float2 *AABBz)
{
    AABBx->x = min(AABB1.x, AABB2.x);
    AABBx->y = max(AABB1.y, AABB2.y);

    AABBy->x = min(AABB1.z, AABB2.z);
    AABBy->y = max(AABB1.w, AABB2.w);

    AABBz->x = min(AABB3.x, AABB3.z);
    AABBz->y = max(AABB3.y, AABB3.w);
}

// Compute the union of two unpacked AABBs.
GRACE_DEVICE void AABB_union(float3 bottom1, float3 top1,
                             float3 bottom2, float3 top2,
                             float3* bottom, float3* top)
{
    bottom->x = min(bottom1.x, bottom2.x);
    top->x    = max(top1.x, top2.x);

    bottom->y = min(bottom1.y, bottom2.y);
    top->y    = max(top1.y, top2.y);

    bottom->z = min(bottom1.z, bottom2.z);
    top->z    = max(top1.z, top2.z);
}

GRACE_DEVICE int2 decode_node(int2 node) {
    int l = node.x;
    int r = node.y;
    return make_int2(-1 * l - 1, -1 * r - 1);
}

GRACE_DEVICE int4 decode_node(int4 node) {
    int2 dec = decode_node(make_int2(node.z, node.w));
    return make_int4(node.x, node.y, dec.x, dec.y);
}

GRACE_DEVICE int2 encode_node(int2 node) {
    int l = node.x;
    int r = node.y;
    return make_int2(-1 * (l + 1), -1 * (r + 1));
}

GRACE_DEVICE int4 encode_node(int4 node) {
    int2 enc = encode_node(make_int2(node.z, node.w));
    return make_int4(node.x, node.y, enc.x, enc.y);
}

GRACE_DEVICE bool is_left_child(int2 node, int parent_index) {
    return node.y == parent_index;
}

GRACE_DEVICE bool is_left_child(int4 node, int parent_index) {
    return node.w == parent_index;
}

// Return the parent index in terms of the base nodes, given the actual parent
// index and the base nodes covered by the current node.
GRACE_DEVICE int logical_parent(int2 b_node, int4 node, int parent_index)
{
    return is_left_child(node, parent_index) ? b_node.y : b_node.x - 1;
}

// Find the parent index of a node.
// node contains the left- and right-most primitives it spans.
// deltas is an array of per-primitive delta values.
template <typename DeltaIter, typename DeltaComp>
GRACE_DEVICE int node_parent(int2 node, DeltaIter deltas, DeltaComp comp)
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
GRACE_DEVICE int node_parent(int4 node, DeltaIter deltas, DeltaComp comp)
{
    return node_parent(make_int2(node.z, node.w), deltas, comp);
}

GRACE_DEVICE int node_size(int2 node)
{
    return node.y - node.x + 1;
}

GRACE_DEVICE int node_size(int4 node)
{
    return node_size(make_int2(node.z, node.w));
}

// Returns true if an inner node at index is out of range for the current block.
GRACE_DEVICE bool out_of_block(int index, int low, int high) {
    return (index < low || index >= high);
}

// Write hierarchy and AABB data to the left of the parent node.
GRACE_DEVICE void propagate_left(int4 node, float3 bottom, float3 top,
                                 int node_index, int parent_index, int4* nodes)
{
    set_left_node(node_index, node.z, parent_index, nodes);
    set_left_AABB(bottom, top, parent_index, nodes);
}

// Write hierarchy and AABB data to the right of the parent node.
GRACE_DEVICE void propagate_right(int4 node, float3 bottom, float3 top,
                                  int node_index, int parent_index, int4* nodes)
{
    set_right_node(node_index, node.w, parent_index, nodes);
    set_right_AABB(bottom, top, parent_index, nodes);
}

template <typename Vec>
struct invalid {};

template <>
struct invalid<int2>
{
    static const int2 node;
};
const int2 invalid<int2>::node = make_int2(-1, -1);

template <>
struct invalid<int4>
{
    static const int4 node;
};
const int4 invalid<int4>::node = make_int4(-1, -1, -1, -1);

template <typename Vec>
struct is_invalid_node : public thrust::unary_function<Vec, bool> {};

template <>
struct is_invalid_node<int4> : public thrust::unary_function<int4, bool>
{
    GRACE_HOST_DEVICE
    bool operator()(const int4 node) const
    {
        return (node.x == -1 && node.y == -1 && node.z == -1 && node.w == -1);
    }
};

template <>
struct is_invalid_node<int2> : public thrust::unary_function<int2, bool>
{
    GRACE_HOST_DEVICE
    bool operator()(const int2 node) const
    {
        return (node.x == -1 && node.y == -1);
    }
};

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
    const int4* leaves,
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
        int4 leaf = leaves[tid];
        int last_idx = leaf.x + leaf.y - 1;
        leaf_deltas[tid] = all_deltas[last_idx];

        tid += blockDim.x * gridDim.x;
    }
}

template <typename DeltaIter, typename DeltaComp>
__global__ void build_leaves_kernel(
    int2* tmp_nodes,
    const size_t n_nodes,
    DeltaIter deltas,
    const int max_per_leaf,
    const DeltaComp delta_comp)
{
    extern __shared__ int SMEM[];

    const size_t n_leaves = n_nodes + 1;

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    ++deltas;

    int bid = blockIdx.x;
    for ( ; bid * grace::BUILD_THREADS_PER_BLOCK < n_leaves; bid += gridDim.x)
    {
        int tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;

        __syncthreads();
        for (int i = threadIdx.x;
             i < grace::BUILD_THREADS_PER_BLOCK + max_per_leaf;
             i += grace::BUILD_THREADS_PER_BLOCK)
        {
            SMEM[i] = -1;
        }
        __syncthreads();

        // [low, high) leaf indices covered by this block, including the
        // max_per_leaf buffer.
        int low = bid * grace::BUILD_THREADS_PER_BLOCK;
        int high = min((bid + 1) * grace::BUILD_THREADS_PER_BLOCK + max_per_leaf,
                       (int)n_leaves);
        // So flags can be accessed directly with a node index.
        int* flags = SMEM - low;

        for (int idx = tid; idx < high; idx += grace::BUILD_THREADS_PER_BLOCK)
        {
            // First node is a leaf at idx.
            int2 node = make_int2(idx, idx);

            bool first_arrival;
            // Climb tree.
            do
            {
                GRACE_ASSERT(node.x >= 0);
                GRACE_ASSERT(node.y > 0 || (node.x == node.y && node.y == 0));
                GRACE_ASSERT(node.x < n_leaves - 1 || (node.x == node.y && node.x == n_leaves - 1));
                GRACE_ASSERT(node.y < n_leaves);

                if (node_size(node) > max_per_leaf) {
                    // Both of this node's children will become wide leaves.
                    break;
                }

                int parent_index = node_parent(node, deltas, delta_comp);

                if (out_of_block(parent_index, low, high)) {
                    break;
                }

                // Propagate left- or right-most primitive up the tree.
                int this_end;
                if (is_left_child(node, parent_index)) {
                    tmp_nodes[parent_index].x = node.x;
                    this_end = node.x;
                }
                else {
                    tmp_nodes[parent_index].y = node.y;
                    this_end = node.y;
                }

                int other_end = atomicExch(flags + parent_index, this_end);
                first_arrival = (other_end == -1);
                GRACE_ASSERT(this_end != other_end);

                node = make_int2(min(this_end, other_end),
                                 max(this_end, other_end));
            } while (!first_arrival);

        } // for idx < high
    } // for bid * grace::BUILD_THREADS_PER_BLOCK < n_leaves
}

__global__ void write_leaves_kernel(
    const int2* nodes,
    const size_t n_nodes,
    int4* big_leaves,
    const int max_per_leaf)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for ( ; tid < n_nodes; tid += blockDim.x * gridDim.x)
    {
        int2 node = nodes[tid];
        GRACE_ASSERT(node.y != 0);

        int2 lchild = make_int2(node.x, tid);
        int2 rchild = make_int2(tid + 1, node.y);

        bool has_left  = (node.x >= 0);
        bool has_right = (node.y >= 0);

        // We only require that (size > max_per_leaf) gives a correct result.
        int size = (has_left && has_right) ? node_size(node) : max_per_leaf + 1;

        if (has_left && size > max_per_leaf) {
            int left = lchild.x;
            int lsize = node_size(lchild);
            big_leaves[left] = make_int4(left, lsize, 0, 0);
        }
        if (has_right && size > max_per_leaf) {
            int right = rchild.x;
            int rsize = node_size(rchild);
            big_leaves[right] = make_int4(right, rsize, 0, 0);
        }
    }
}

__global__ void fix_leaf_ranges(int4* leaves, const size_t n_leaves)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for ( ; tid < n_leaves; tid += blockDim.x * gridDim.x)
    {
        int4 leaf = leaves[tid];
        leaf.z = leaf.w = tid;
        leaves[tid] = leaf;
    }
}

template <
    typename PrimitiveIter,
    typename DeltaIter,
    typename DeltaComp,
    typename AABBFunc
    >
__global__ void build_nodes_slice_kernel(
    int4* nodes,
    float4* f4_nodes,
    const size_t n_nodes,
    const int4* leaves,
    const size_t n_leaves,
    PrimitiveIter primitives,
    const int* base_indices,
    const size_t n_base_nodes,
    int* root_index,
    DeltaIter deltas,
    const int max_per_node,
    int* new_base_indices,
    const DeltaComp delta_comp,
    const AABBFunc AABB)
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;

    __shared__ float3 AABB_min[grace::BUILD_THREADS_PER_BLOCK];
    __shared__ float3 AABB_max[grace::BUILD_THREADS_PER_BLOCK];
    __shared__ int g_other_end[grace::BUILD_THREADS_PER_BLOCK];
    __shared__ int b_other_end[grace::BUILD_THREADS_PER_BLOCK];
    extern __shared__ int SMEM[];

    int* sm_flags = SMEM;

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    ++deltas;

    int bid = blockIdx.x;
    for ( ; bid * grace::BUILD_THREADS_PER_BLOCK < n_base_nodes; bid += gridDim.x)
    {
        int tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;

        __syncthreads();
        for (int i = threadIdx.x;
             i < grace::BUILD_THREADS_PER_BLOCK + max_per_node;
             i += grace::BUILD_THREADS_PER_BLOCK)
        {
            sm_flags[i] = -1;
        }
        __syncthreads();

        // high and high2 differ only for the final block, where the range of
        // nodes which may be climbed to [low, high) is one less than the range
        // of input base nodes, [low, high2).
        int low = bid * grace::BUILD_THREADS_PER_BLOCK;
        int high = min((bid + 1) * grace::BUILD_THREADS_PER_BLOCK + max_per_node,
                       (int)(n_base_nodes - 1));
        int high2 = (high == (int)(n_base_nodes - 1) ? high + 1 : high);

        // So b_nodes and flags can be accessed directly with a (logical) node
        // index.
        int* flags = sm_flags - low;

        for (int idx = tid; idx < high2; idx += grace::BUILD_THREADS_PER_BLOCK)
        {
            // For the tree climb, we start at base nodes, treating them as
            // leaves.  The left/right values refer to the left- and right-most
            // actual leaves covered by the current node.  (b_left and b_right
            // contain the left- and right-most base node indices.)
            // b_index is a logical node index, and is used for writing to
            // shared memory.
            int b_index = idx;
            // These are real node indices (for writing to global memory).
            int g_index = base_indices[b_index];

            // Node index can be >= n_nodes if a leaf.
            GRACE_ASSERT(g_index < n_nodes + n_leaves);

            int4 g_node = get_node(g_index, nodes, leaves, n_nodes);
            // b_node contains left- and right-most base node indices.
            int2 b_node = make_int2(b_index, b_index);

            // Base nodes can be inner nodes or leaves.
            GRACE_ASSERT(g_node.x >= 0);
            GRACE_ASSERT(g_node.y > 0);
            GRACE_ASSERT(g_node.x < n_nodes + n_leaves - 1 || is_leaf(g_index, n_nodes));
            GRACE_ASSERT(g_node.y <= n_nodes + n_leaves - 1);
            GRACE_ASSERT(g_node.z >= 0);
            GRACE_ASSERT(g_node.w > 0 || (is_leaf(g_index, n_nodes) && g_node.w == 0));
            GRACE_ASSERT(g_node.z < n_leaves - 1 || (is_leaf(g_index, n_nodes) && g_node.z == n_leaves - 1));
            GRACE_ASSERT(g_node.w <= n_leaves - 1);

            int g_parent_index = node_parent(g_node, deltas, delta_comp);
            int b_parent_index = logical_parent(b_node, g_node, g_parent_index);

            if (out_of_block(b_parent_index, low, high)) {
                continue;
            }

            float3 bot, top;
            if (!is_leaf(g_index, n_nodes)) {
                float4 AABB1 = get_AABB1(g_index, f4_nodes);
                float4 AABB2 = get_AABB2(g_index, f4_nodes);
                float4 AABB3 = get_AABB3(g_index, f4_nodes);

                AABB_union(AABB1, AABB2, AABB3, &bot, &top);
            }
            else {
                bot.x = bot.y = bot.z = CUDART_INF_F;
                top.x = top.y = top.z = -1.f;

                #pragma unroll 4
                for (int i = 0; i < g_node.y; i++) {
                    TPrimitive prim = primitives[g_node.x + i];

                    float3 pbot, ptop;
                    AABB(prim, &pbot, &ptop);

                    AABB_union(bot, top, pbot, ptop, &bot, &top);
                }
            }

            // Note, they should never be equal.
            GRACE_ASSERT(bot.x < top.x);
            GRACE_ASSERT(bot.y < top.y);
            GRACE_ASSERT(bot.z < top.z);

            AABB_min[threadIdx.x] = bot;
            AABB_max[threadIdx.x] = top;

            if (is_left_child(g_node, g_parent_index))
            {
                propagate_left(g_node, bot, top, g_index, g_parent_index,
                               nodes);
                g_other_end[threadIdx.x] = g_node.z;
                b_other_end[threadIdx.x] = b_node.x;
            }
            else
            {
                propagate_right(g_node, bot, top, g_index, g_parent_index,
                                nodes);
                g_other_end[threadIdx.x] = g_node.w;
                b_other_end[threadIdx.x] = b_node.y;
            }

            // Travel up the tree.  The second thread to reach a node writes its
            // logical/compressed left or right end to its parent; the first
            // exits the loop.
            __threadfence_block();
            int other_idx = atomicExch(flags + b_parent_index, threadIdx.x);
            GRACE_ASSERT(other_idx != threadIdx.x);

            bool first_arrival = (other_idx == -1);
            while (!first_arrival)
            {
                int g_end = g_other_end[other_idx];
                int b_end = b_other_end[other_idx];
                if (is_left_child(g_node, g_parent_index)) {
                    g_node.w = g_end;
                    b_node.y = b_end;
                }
                else {
                    g_node.z = g_end;
                    b_node.x = b_end;
                }
                // Now b_node.xy are correct, g_node.zw are correct.

                b_index = b_parent_index;
                g_index = g_parent_index;

                GRACE_ASSERT(b_index < n_base_nodes);
                GRACE_ASSERT(g_index < n_nodes);

                GRACE_ASSERT(g_node.z >= 0);
                GRACE_ASSERT(g_node.w > 0);
                GRACE_ASSERT(g_node.z < n_leaves - 1);
                GRACE_ASSERT(g_node.w <= n_leaves - 1);

                // We are the second thread in this block to reach this node.
                // Both of its logical/compressed end-indices must also be in
                // this block.
                GRACE_ASSERT(b_node.x >= low);
                GRACE_ASSERT(b_node.y > low);
                GRACE_ASSERT(b_node.x < high2 - 1);
                GRACE_ASSERT(b_node.y < high2);

                g_parent_index = node_parent(g_node, deltas, delta_comp);
                b_parent_index = logical_parent(b_node, g_node, g_parent_index);

                // Even if this is true, the following size test can be false.
                if (node_size(g_node) == n_leaves) {
                    *root_index = g_index;
                }

                // Exit the loop if at least one of our child nodes is large
                // enough to become a base node.
                if (node_size(b_node) > max_per_node) {
                    break;
                }

                if (out_of_block(b_parent_index, low, high)) {
                    // Parent node outside this block's boundaries.  Either a
                    // thread in an adjacent block will follow this path, or the
                    // current node will be a base node in the next iteration.
                    break;
                }

                float3 other_bot = AABB_min[other_idx];
                float3 other_top = AABB_max[other_idx];
                AABB_union(bot, top, other_bot, other_top, &bot, &top);

                GRACE_ASSERT(bot.x < top.x);
                GRACE_ASSERT(bot.y < top.y);
                GRACE_ASSERT(bot.z < top.z);

                AABB_min[threadIdx.x] = bot;
                AABB_max[threadIdx.x] = top;

                if (is_left_child(g_node, g_parent_index))
                {
                    propagate_left(g_node, bot, top, g_index, g_parent_index,
                                   nodes);
                    g_other_end[threadIdx.x] = g_node.z;
                    b_other_end[threadIdx.x] = b_node.x;
                }
                else
                {
                    propagate_right(g_node, bot, top, g_index, g_parent_index,
                                    nodes);
                    g_other_end[threadIdx.x] = g_node.w;
                    b_other_end[threadIdx.x] = b_node.y;
                }

                __threadfence_block();
                other_idx = atomicExch(flags + b_parent_index, threadIdx.x);
                GRACE_ASSERT(other_idx != threadIdx.x);
                first_arrival = (other_idx == -1);
            } // while (!first_arrival)
        } // for idx < high
    } // for bid * BUILD_THREADS_PER_BLOCK < n_base_nodes
    return;
}

template <typename DeltaIter, typename DeltaComp>
__global__ void fill_output_queue(
    const int4* nodes,
    const size_t n_nodes,
    const int4* leaves,
    DeltaIter deltas,
    const int* old_base_indices,
    const size_t n_base_indices,
    int* new_base_indices,
    DeltaComp delta_comp)
{
    int tid = threadIdx.x + blockIdx.x * grace::BUILD_THREADS_PER_BLOCK;

    // So deltas[-1] is a valid index.
    ++deltas;

    for ( ; tid < n_base_indices; tid += blockDim.x * gridDim.x)
    {
        int node_index = old_base_indices[tid];
        int4 node = get_node(node_index, nodes, leaves, n_nodes);

        bool climb = true;
        while (climb)
        {
            GRACE_ASSERT(node.x >= 0);
            GRACE_ASSERT(node.y > 0);
            GRACE_ASSERT(node.x < 2 * n_nodes || is_leaf(node_index, n_nodes));
            GRACE_ASSERT(node.y <= 2 * n_nodes);
            GRACE_ASSERT(node.z >= 0);
            GRACE_ASSERT(node.w > 0 || (is_leaf(node_index, n_nodes) && node.w == 0));
            GRACE_ASSERT(node.z < n_nodes || (is_leaf(node_index, n_nodes) && node.z == n_nodes));
            GRACE_ASSERT(node.w <= n_nodes);

            int child_index = node_index;
            int4 child = node;

            node_index = node_parent(child, deltas, delta_comp);
            node = get_inner(node_index, nodes);

            // Nodes written to have valid data.
            int children = (node.x >= 0) + (node.y >= 0);

            // If the child we came from was the _only_ child written to the
            // current node (children == 1), that child should be added to the
            // work queue.
            // If the child we came from was _not_ written to the current node
            // (children == 0 or children == 1), that child should be added
            // to the work queue.
            // Otherwise, we should continue the climb.
            if (children < 2) {
                GRACE_ASSERT(new_base_indices[tid] == -1);
                // Write child node.
                // Only one thread may reach this point, and writes its child.
                new_base_indices[tid] = child_index;
                climb = false;
            }

            // Thread coming from left child is allowed to continue.
            climb = climb && (child_index == node.x);
        }
    }
}

//-----------------------------------------------------------------------------
// C-like wrappers for ALBVH kernels.
//-----------------------------------------------------------------------------

// Two template parameters as DeltaIter _may_ be const_iterator or const T*.
template<typename DeltaIter, typename LeafDeltaIter>
GRACE_HOST void copy_leaf_deltas(
    const thrust::device_vector<int4>& d_leaves,
    DeltaIter d_all_deltas_iter,
    LeafDeltaIter d_leaf_deltas_iter)
{
    const int blocks = min(grace::MAX_BLOCKS,
                           static_cast<int>((d_leaves.size() + 511) / 512 ));
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
    thrust::device_vector<int4>& d_leaves,
    const int max_per_leaf,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp)
{
    const size_t n_leaves = d_leaves.size();
    const size_t n_nodes = n_leaves - 1;
    GRACE_ASSERT(n_nodes <= d_tmp_nodes.size());

    if (n_leaves <= max_per_leaf) {
        const std::string msg
            = "max_per_leaf must be less than the total number of primitives.";
        throw std::invalid_argument(msg);
    }

    thrust::fill(d_leaves.begin(), d_leaves.end(), invalid<int4>::node);
    thrust::fill(d_tmp_nodes.begin(), d_tmp_nodes.end(), invalid<int2>::node);

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_leaves + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));
    int smem_size = sizeof(int) * (grace::BUILD_THREADS_PER_BLOCK + max_per_leaf);

    build_leaves_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK, smem_size>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        d_deltas_iter,
        max_per_leaf,
        delta_comp);
    GRACE_KERNEL_CHECK();

    blocks = min(grace::MAX_BLOCKS,
                 (int) ((n_nodes + grace::BUILD_THREADS_PER_BLOCK - 1)
                         / grace::BUILD_THREADS_PER_BLOCK));

    write_leaves_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_leaves.data()),
        max_per_leaf);
    GRACE_KERNEL_CHECK();

    // Using
    //   const invalid4 = make_int4(-1, -1, -1, -1);
    //   thrust::remove(..., invalid4);
    // gives:
    //   ... error: no operator "==" matches these operands
    //           operand types are: const in4 == const int4
    d_leaves.erase(thrust::remove_if(d_leaves.begin(), d_leaves.end(),
                                     is_invalid_node<int4>()),
                   d_leaves.end());

    blocks = min(grace::MAX_BLOCKS,
                 (int) ((d_leaves.size() + grace::BUILD_THREADS_PER_BLOCK - 1)
                         / grace::BUILD_THREADS_PER_BLOCK));

    fix_leaf_ranges<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_leaves.data()),
        d_leaves.size()
    );
}

template <
    typename PrimitiveIter,
    typename DeltaIter,
    typename DeltaComp,
    typename AABBFunc
    >
GRACE_HOST void build_nodes(
    Tree& d_tree,
    PrimitiveIter d_prims_iter,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp,
    const AABBFunc AABB)
{
    const size_t n_leaves = d_tree.leaves.size();
    const size_t n_nodes = n_leaves - 1;

    thrust::fill(d_tree.nodes.begin(), d_tree.nodes.end(), invalid<int4>::node);

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

        // Output queue is always filled with invalid values so we can remove
        // them and use it as input in the next iteration.
        thrust::fill(out_q_begin, out_q_end, -1);

        int blocks = min(grace::MAX_BLOCKS,
                         (int) ((n_in + grace::BUILD_THREADS_PER_BLOCK - 1)
                                 / grace::BUILD_THREADS_PER_BLOCK));
        // SMEM has to cover for BUILD_THREADS_PER_BLOCK + max_per_leaf flags
        // AND int2 nodes.
        int smem_size = (sizeof(int) + sizeof(int2))
                        * (grace::BUILD_THREADS_PER_BLOCK + d_tree.max_per_leaf);
        build_nodes_slice_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK, smem_size>>>(
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            reinterpret_cast<float4*>(
                thrust::raw_pointer_cast(d_tree.nodes.data())),
            n_nodes,
            thrust::raw_pointer_cast(d_tree.leaves.data()),
            n_leaves,
            d_prims_iter,
            d_in_ptr,
            n_in,
            d_tree.root_index_ptr,
            d_deltas_iter,
            d_tree.max_per_leaf, // This can actually be anything.
            d_out_ptr,
            delta_comp,
            AABB);
        GRACE_KERNEL_CHECK();

        blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_in + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));
        fill_output_queue<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            n_nodes,
            thrust::raw_pointer_cast(d_tree.leaves.data()),
            d_deltas_iter,
            d_in_ptr,
            n_in,
            d_out_ptr,
            delta_comp);
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
}

} // namespace ALBVH


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
    int blocks = min(grace::MAX_BLOCKS, (int)((N_deltas + 512 - 1) / 512));
    ALBVH::compute_deltas_kernel<<<blocks, 512>>>(
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
    Tree& d_tree,
    PrimitiveIter d_prims_iter,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp,
    const AABBFunc AABB)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;

    // In case this ever changes.
    GRACE_ASSERT(sizeof(int4) == sizeof(float4));

    const size_t n_leaves = d_tree.leaves.size();
    const size_t n_nodes = n_leaves - 1;

    thrust::device_vector<int2> d_tmp_nodes(n_nodes);
    ALBVH::build_leaves(d_tmp_nodes, d_tree.leaves, d_tree.max_per_leaf,
                        d_deltas_iter, delta_comp);

    const size_t n_new_leaves = d_tree.leaves.size();
    const size_t n_new_nodes = n_new_leaves - 1;

    d_tree.nodes.resize(4 * n_new_nodes);

    thrust::device_vector<DeltaType> d_new_deltas(n_new_leaves + 1);
    DeltaType* new_deltas_ptr = thrust::raw_pointer_cast(d_new_deltas.data());

    ALBVH::copy_leaf_deltas(d_tree.leaves, d_deltas_iter, new_deltas_ptr);
    ALBVH::build_nodes(d_tree, d_prims_iter, new_deltas_ptr, delta_comp, AABB);
}

template <
    typename TPrimitive,
    typename DeltaType,
    typename DeltaComp,
    typename AABBFunc
    >
GRACE_HOST void build_ALBVH(
    Tree& d_tree,
    const thrust::device_vector<TPrimitive>& d_primitives,
    const thrust::device_vector<DeltaType>& d_deltas,
    const DeltaComp delta_comp,
    const AABBFunc AABB)
{
    const TPrimitive* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    const DeltaType* deltas_ptr = thrust::raw_pointer_cast(d_deltas.data());

    build_ALBVH(d_tree, prims_ptr, deltas_ptr, delta_comp, AABB);
}

// Specialized with DeltaComp = thrust::less<DeltaType>
template <typename PrimitiveIter, typename DeltaIter, typename AABBFunc>
GRACE_HOST void build_ALBVH(
    Tree& d_tree,
    PrimitiveIter d_prims_iter,
    DeltaIter d_deltas_iter,
    const AABBFunc AABB)
{
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;
    typedef typename thrust::less<DeltaType> DeltaComp;

    build_ALBVH(d_tree, d_prims_iter, d_deltas_iter, DeltaComp(), AABB);
}

// Specialized with DeltaComp = thrust::less<DeltaType>
template <typename TPrimitive, typename DeltaType, typename AABBFunc>
GRACE_HOST void build_ALBVH(
    Tree& d_tree,
    const thrust::device_vector<TPrimitive>& d_primitives,
    const thrust::device_vector<DeltaType>& d_deltas,
    const AABBFunc AABB)
{
    typedef typename thrust::less<DeltaType> DeltaComp;
    const TPrimitive* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    const DeltaType* deltas_ptr = thrust::raw_pointer_cast(d_deltas.data());

    build_ALBVH(d_tree, prims_ptr, deltas_ptr, DeltaComp(), AABB);
}

} // namespace grace
