#pragma once

// CUDA math constants.
#include <math_constants.h>

#include <climits>
#include <stdexcept>
#include <string>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/swap.h>
#include <thrust/transform.h>

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
GRACE_DEVICE int logical_parent(int4 node, int2 g_node, int g_parent_index)
{
    return is_left_child(g_node, g_parent_index) ? node.w : node.z - 1;
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

// Return the index of the largest delta-value in the range [begin, end).
template <typename DeltaIter, typename DeltaComp>
GRACE_HOST_DEVICE int find_split(int begin, int end, DeltaIter deltas,
                                 DeltaComp comp)
{
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;

    int max_delta_idx = begin;
    DeltaType max_delta = deltas[begin];

    for (int i = begin + 1; i < end; ++i)
    {
        DeltaType delta = deltas[i];

        if (comp(max_delta, delta)) {
            max_delta = delta;
            max_delta_idx = i;
        }
    }

    return max_delta_idx;
}

// TODO: This is just a segmented reduction over deltas. Use e.g. MGPU.
//
// Note: tid == n_segments - 1 should always find n_primitives - 1 for the
// split index. This is not really the desired behaviour, as it may be that
// the final segment should would be better if split into < max_per_group
// sub-segments. But this only affects the final wide leaf, so it's essentially
// inconsequential.
template <typename DeltaIter, typename DeltaComp>
__global__ void coarse_splits_kernel(
    const size_t n_primitives,
    const int max_per_group,
    DeltaIter deltas,
    int* splits,
    const size_t n_segments,
    DeltaComp delta_comp)
{
    // Because deltas[0] == delta_function(-1, 0).
    ++deltas;

    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < n_segments;
         tid += blockDim.x * gridDim.x)
    {
        const int begin = tid * max_per_group;
        const int end = min((tid + 1) * max_per_group,
                            static_cast<int>(n_primitives));

        splits[tid] = find_split(begin, end, deltas, delta_comp);
    }
}

template <typename DeltaIter, typename DeltaComp>
__global__ void count_fine_splits_kernel(
    const int max_per_group,
    DeltaIter deltas,
    const int* splits,
    const size_t n_splits,
    int* split_counts,
    DeltaComp delta_comp)
{
    // So deltas[-1] is valid.
    ++deltas;

    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < n_splits;
         tid += blockDim.x * gridDim.x)
    {
        int begin = (tid > 0) ? splits[tid - 1] + 1 : 0;
        int end = splits[tid];
        const int size = end - begin + 1;

        // We always have at least one (coarse) split.
        int count = 1;
        if (size > max_per_group)
        {
            GRACE_ASSERT(size < 2 * max_per_group);

            do
            {
                ++count;

                const int split_idx = find_split(begin, end, deltas, delta_comp);
                const int lsize = split_idx - begin + 1;
                const int rsize = end - split_idx;

                GRACE_ASSERT(!(lsize > max_per_group && rsize > max_per_group));

                begin = (lsize > max_per_group) ? begin : split_idx + 1;
                end   = (rsize > max_per_group) ? end   : split_idx;
            }
            while (begin < end);
        }

        split_counts[tid] = count;
    }
}

template <typename DeltaIter, typename DeltaComp>
__global__ void fine_splits_kernel(
    const int max_per_group,
    DeltaIter deltas,
    const int* split_offsets,
    const size_t n_coarse_splits,
    int* fine_splits,
    DeltaComp delta_comp)
{
    // So deltas[-1] is valid.
    ++deltas;

    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < n_coarse_splits;
         tid += blockDim.x * gridDim.x)
    {
        const int begin_offset = (tid > 0) ? split_offsets[tid - 1] : 0;
        const int end_offset = split_offsets[tid];

        int begin = (tid > 0) ? fine_splits[begin_offset] + 1 : 0;
        int end = fine_splits[end_offset];
        const int size = end - begin + 1;

        if (size > max_per_group)
        {
            GRACE_ASSERT(size < 2 * max_per_group);
            GRACE_ASSERT(size > 0);

            // Splits must be output in ascending order (of split index).
            // We ensure this by writing either the first- or last-most split
            // each time, updating first/last after each write.
            int first = begin_offset + 1;
            int last  = end_offset - 1;

            do
            {
                GRACE_ASSERT(first <= last);

                const int split_idx = find_split(begin, end, deltas, delta_comp);
                const int lsize = split_idx - begin + 1;
                const int rsize = end - split_idx;

                GRACE_ASSERT(!(lsize > max_per_group && rsize > max_per_group));
                GRACE_ASSERT(lsize > 0);
                GRACE_ASSERT(rsize > 0);

                // For the final split. This has to be checked first!
                if (first == last) fine_splits[first] = split_idx;

                if (lsize > max_per_group) fine_splits[last--] = split_idx;
                if (rsize > max_per_group) fine_splits[first++] = split_idx;

                begin = (lsize > max_per_group) ? begin : split_idx + 1;
                end   = (rsize > max_per_group) ? end   : split_idx;
            }
            while (begin < end);
        }
    }
}

__global__ void write_leaves_kernel(
    const int* splits,
    const size_t n_splits,
    const int max_per_leaf,
    int4* wide_leaves)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < n_splits;
         tid += blockDim.x * gridDim.x)
    {
        const int begin = (tid > 0) ? splits[tid - 1] + 1 : 0;
        const int end = splits[tid];
        const int size = end - begin + 1;

        GRACE_ASSERT(size <= max_per_leaf);
        GRACE_ASSERT(size > 0);

        const int4 leaf = make_int4(begin, size, tid, tid);
        wide_leaves[tid] = leaf;
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
    extern __shared__ int SMEM[];

    int* sm_flags = SMEM;
    volatile int2* sm_nodes
        = (int2*)&sm_flags[grace::BUILD_THREADS_PER_BLOCK + max_per_node];

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
            sm_flags[i] = 0;
            sm_nodes[i].x = 0;
            sm_nodes[i].y = 0;
        }
        __syncthreads();

        // The base-node indices for this block cover the range [low, high),
        // including the max_per_node buffer.
        int low = bid * grace::BUILD_THREADS_PER_BLOCK;
        int high = min((bid + 1) * grace::BUILD_THREADS_PER_BLOCK + max_per_node,
                       (int)n_base_nodes);
        // So g_nodes and flags can be accessed directly with a (logical) node
        // index.
        int* flags = sm_flags - low;
        volatile int2* g_nodes = sm_nodes - low;

        for (int idx = tid; idx < high; idx += grace::BUILD_THREADS_PER_BLOCK)
        {
            // For the tree climb, we start at base nodes, treating them as
            // leaves.  The left/right values refer to the left- and right-most
            // *base nodes* covered by the current node.  (g_left and g_right
            // contain the left- and right-most *actual* leaf indices.)
            // The current index (i.e. the idx-th base node) is thus a logical
            // node index, and is used for writing to shared memory.
            int cur_index = idx;
            // These are real node indices (for writing to global memory).
            int g_cur_index = base_indices[cur_index];

            // Node index can be >= n_nodes if a leaf.
            GRACE_ASSERT(g_cur_index < n_nodes + n_leaves);

            int4 node = get_node(g_cur_index, nodes, leaves, n_nodes);
            // g_node left- and right-most leaf node incides.
            int2 g_node = make_int2(node.z, node.w);
            // node contains left- and right-most base node indices.
            node.z = node.w = cur_index;

            // Base nodes can be inner nodes or leaves.
            GRACE_ASSERT(g_node.x >= 0);
            GRACE_ASSERT(g_node.y > 0 || (g_node.x == g_node.y && g_node.y == 0));
            GRACE_ASSERT(g_node.x < n_leaves - 1 || (g_node.x == g_node.y && g_node.x == n_leaves - 1));
            GRACE_ASSERT(g_node.y < n_leaves);

            int g_parent_index = node_parent(g_node, deltas, delta_comp);
            int parent_index = logical_parent(node, g_node, g_parent_index);

            if (out_of_block(parent_index, low, high)) {
                continue;
            }

            float3 bot, top;
            if (g_cur_index < n_nodes) {
                float4 AABB1 = get_AABB1(g_cur_index, f4_nodes);
                float4 AABB2 = get_AABB2(g_cur_index, f4_nodes);
                float4 AABB3 = get_AABB3(g_cur_index, f4_nodes);

                AABB_union(AABB1, AABB2, AABB3, &bot, &top);
            }
            else {
                bot.x = bot.y = bot.z = CUDART_INF_F;
                top.x = top.y = top.z = -1.f;

                #pragma unroll 4
                for (int i = 0; i < node.y; i++) {
                    TPrimitive prim = primitives[node.x + i];

                    float3 pbot, ptop;
                    AABB(prim, &pbot, &ptop);

                    AABB_union(bot, top, pbot, ptop, &bot, &top);
                }
            }

            // Note, they should never be equal.
            GRACE_ASSERT(bot.x < top.x);
            GRACE_ASSERT(bot.y < top.y);
            GRACE_ASSERT(bot.z < top.z);

            // Encoding identifies that the node was written this iteration.
            node = encode_node(node);
            if (is_left_child(g_node, g_parent_index))
            {
                propagate_left(node, bot, top, g_cur_index, g_parent_index,
                               nodes);
                g_nodes[parent_index].x = g_node.x;
            }
            else
            {
                propagate_right(node, bot, top, g_cur_index, g_parent_index,
                                nodes);
                g_nodes[parent_index].y = g_node.y;
            }

            // Travel up the tree.  The second thread to reach a node writes its
            // logical/compressed left or right end to its parent; the first
            // exits the loop.
            __threadfence_block();
            unsigned int count = atomicAdd(flags + parent_index, 1);
            GRACE_ASSERT(count < 2);

            bool first_arrival = (count == 0);
            while (!first_arrival)
            {
                cur_index = parent_index;
                g_cur_index = g_parent_index;

                GRACE_ASSERT(cur_index < n_base_nodes);
                GRACE_ASSERT(g_cur_index < n_nodes);

                // We are certain that a thread in this block has already
                // *written* the other child of the current node, so we can read
                // from L1 if we get a cache hit.
                // int4 node = load_vec4s32(&(nodes[4 * g_cur_index + 0].x));
                int4 node = get_inner(g_cur_index, nodes);

                // 'error: class "int2" has no suitable copy constructor'
                // int2 left_right = g_nodes[cur_index];
                // int g_left = left_right.x;
                // int g_right = left_right.y;
                int g_left = g_nodes[cur_index].x;
                int g_right = g_nodes[cur_index].y;
                int2 g_node = make_int2(g_left, g_right);
                GRACE_ASSERT(g_node.x >= 0);
                GRACE_ASSERT(g_node.y > 0);
                GRACE_ASSERT(g_node.x < n_leaves - 1);
                GRACE_ASSERT(g_node.y < n_leaves);

                // Undo the -ve sign encoding.
                node = decode_node(node);
                // We are the second thread in this block to reach this node.
                // Both of its logical/compressed end-indices must also be in
                // this block.
                GRACE_ASSERT(node.z >= low);
                GRACE_ASSERT(node.w > low);
                GRACE_ASSERT(node.z < high - 1);
                GRACE_ASSERT(node.w < high);

                g_parent_index = node_parent(g_node, deltas, delta_comp);
                parent_index = logical_parent(node, g_node, g_parent_index);

                // Even if this is true, the following compacted/logical size
                // test can be false.
                if (node_size(g_node) == n_leaves) {
                    *root_index = g_cur_index;
                }

                // Check our compacted/logical size, and exit the loop if we are
                // large enough to become a base layer in the next iteration.
                if (node_size(node) > max_per_node) {
                    // Both children of the current node must be leaves.  Stop
                    // travelling up the tree and continue with the outer loop.
                    break;
                }

                if (out_of_block(parent_index, low, high)) {
                    // Parent node outside this block's boundaries.  Either a
                    // thread in an adjacent block will follow this path, or the
                    // current node will be a base node in the next iteration.
                    break;
                }

                float4 AABB1 = get_AABB1(g_cur_index, f4_nodes);
                float4 AABB2 = get_AABB2(g_cur_index, f4_nodes);
                float4 AABB3 = get_AABB3(g_cur_index, f4_nodes);

                float3 bot, top;
                AABB_union(AABB1, AABB2, AABB3, &bot, &top);
                GRACE_ASSERT(bot.x < top.x);
                GRACE_ASSERT(bot.y < top.y);
                GRACE_ASSERT(bot.z < top.z);

                node = encode_node(node);
                if (is_left_child(g_node, g_parent_index))
                {
                    propagate_left(node, bot, top, g_cur_index, g_parent_index,
                                   nodes);
                    g_nodes[parent_index].x = g_node.x;
                }
                else
                {
                    propagate_right(node, bot, top, g_cur_index, g_parent_index,
                                    nodes);
                    g_nodes[parent_index].y = g_node.y;
                }

                __threadfence_block();
                unsigned int count = atomicAdd(flags + parent_index, 1);
                GRACE_ASSERT(count < 2);
                first_arrival = (count == 0);
            } // while (!first_arrival)
        } // for idx < high
    } // for bid * BUILD_THREADS_PER_BLOCK < n_base_nodes
    return;
}

__global__ void fill_output_queue(
    const int4* nodes,
    const size_t n_nodes,
    const int max_per_node,
    int* new_base_indices)
{
    int tid = threadIdx.x + blockIdx.x * grace::BUILD_THREADS_PER_BLOCK;

    for ( ; tid < n_nodes; tid += blockDim.x * gridDim.x)
    {
        int4 node = get_inner(tid, nodes);

        // The left/right values are negative IFF they were written in this
        // iteration.  A negative index represents a logical/compacted index.
        bool left_written = (node.z < 0);
        bool right_written = (node.w < 0);

        // Undo the -ve encoding.
        node = decode_node(node);
        int left = node.z;
        int right = node.w;

        if (left_written != right_written) {
            // Only one child was written; it must be placed in the output queue
            // at a *unique* location.
            int index = left_written ? left : right;
            GRACE_ASSERT(new_base_indices[index] == -1);
            new_base_indices[index] = left_written ? node.x : node.y;
        }
        else if (left_written) {
            // Both were written, so the current node must be added to the
            // output queue IFF it is sufficiently large; that is, if its size
            // is such that it would have caused the thread to end its tree
            // climb.
            // Again, the location we write to must be unique.
            int size = node_size(node);
            if (size > max_per_node) {
                GRACE_ASSERT(new_base_indices[left] == -1);
                new_base_indices[left] = tid;
            }
        }
    }
}

__global__ void fix_node_ranges(
    int4* nodes,
    const size_t n_nodes,
    const int4* leaves,
    const int* old_base_indices)
{
    int tid = threadIdx.x + blockIdx.x * grace::BUILD_THREADS_PER_BLOCK;

    for ( ; tid < n_nodes; tid += blockDim.x * gridDim.x)
    {
        int4 node = get_inner(tid, nodes);

        // We only need to fix nodes whose ranges were written, *in full*, this
        // iteration.
        if (node.z < 0 && node.w < 0)
        {
            node = decode_node(node);
            int left = node.z;
            int right = node.w;

            // All base nodes have correct range indices.  If we know our
            // left/right-most base node, we can therefore find our
            // left/right-most leaf indices.
            // Note that for leaves, the left and right ranges are simply the
            // (corrected) index of the leaf.
            int index = old_base_indices[left];
            if (is_leaf(index, n_nodes))
                left = index - n_nodes;
            else
                left = get_inner(index, nodes).z;

            index = old_base_indices[right];
            if (is_leaf(index, n_nodes))
                right = index - n_nodes;
            else
                right = get_inner(index, nodes).w;

            // Only the left-most leaf can have index 0.
            GRACE_ASSERT(left >= 0);
            GRACE_ASSERT(left < n_nodes);
            GRACE_ASSERT(right > 0);
            // Only the right-most leaf can have index n_leaves - 1 == n_nodes.
            GRACE_ASSERT(right <= n_nodes);

            node.z = left;
            node.w = right;
            set_inner(node, tid, nodes);
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
    Tree& d_tree,
    const size_t n_primitives,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp)
{
    const int max_per_leaf = d_tree.max_per_leaf;

    if (n_primitives <= max_per_leaf) {
        const std::string msg
            = "max_per_leaf must be less than the total number of primitives.";
        throw std::invalid_argument(msg);
    }

    const size_t n_segments = (n_primitives + max_per_leaf - 1) / max_per_leaf;
    thrust::device_vector<int> d_coarse_splits(n_segments);
    thrust::device_vector<int> d_split_counts(n_segments);

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_segments + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));
    coarse_splits_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        n_primitives,
        max_per_leaf,
        d_deltas_iter,
        thrust::raw_pointer_cast(d_coarse_splits.data()),
        n_segments,
        delta_comp);
    GRACE_KERNEL_CHECK();

    count_fine_splits_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        max_per_leaf,
        d_deltas_iter,
        thrust::raw_pointer_cast(d_coarse_splits.data()),
        n_segments,
        thrust::raw_pointer_cast(d_split_counts.data()),
        delta_comp);
    GRACE_KERNEL_CHECK();

    thrust::inclusive_scan(d_split_counts.begin(), d_split_counts.end(),
                           d_split_counts.begin());
    const int total_splits = d_split_counts.back();

    // Scatter the coarse splits we already have into the array of fine splits.
    // Because we used an inclusive scan, the scatter operation must operate as
    //     d_fine_splits[d_split_counts[i] - 1] = d_coarse_splits[i],
    // fine_splits_kernel needs a similarly modified d_fine_splits array.
    // It is easiest to just modify the array.
    // (This is safe because d_split_counts[i] >= 1 for all i.)
    {
        using namespace thrust::placeholders;
        thrust::transform(d_split_counts.begin(), d_split_counts.end(),
                          d_split_counts.begin(), _1 - 1);
    }
    thrust::device_vector<int> d_fine_splits(total_splits);
    thrust::scatter(d_coarse_splits.begin(), d_coarse_splits.end(),
                    d_split_counts.begin(),
                    d_fine_splits.begin());

    fine_splits_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        max_per_leaf,
        d_deltas_iter,
        thrust::raw_pointer_cast(d_split_counts.data()),
        d_coarse_splits.size(),
        thrust::raw_pointer_cast(d_fine_splits.data()),
        delta_comp);
    GRACE_KERNEL_CHECK();

    d_tree.leaves.resize(total_splits);
    d_tree.nodes.resize(4 * (total_splits - 1));

    write_leaves_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_fine_splits.data()),
        d_fine_splits.size(),
        max_per_leaf,
        thrust::raw_pointer_cast(d_tree.leaves.data()));
    GRACE_KERNEL_CHECK();
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
                     (int) ((n_nodes + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));
        fill_output_queue<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            n_nodes,
            d_tree.max_per_leaf,
            d_out_ptr);
        GRACE_KERNEL_CHECK();

        fix_node_ranges<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            n_nodes,
            thrust::raw_pointer_cast(d_tree.leaves.data()),
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
    const AABBFunc AABB,
    const bool wipe = false)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.

    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;

    // In case this ever changes.
    GRACE_ASSERT(sizeof(int4) == sizeof(float4));

    if (wipe) {
        int4 empty = make_int4(0, 0, 0, 0);
        thrust::fill(d_tree.nodes.begin(), d_tree.nodes.end(), empty);
        thrust::fill(d_tree.leaves.begin(), d_tree.leaves.end(), empty);
    }

    const size_t n_leaves = d_tree.leaves.size();
    const size_t n_nodes = n_leaves - 1;
    thrust::device_vector<int2> d_tmp_nodes(n_nodes);

    ALBVH::build_leaves(d_tree, n_leaves,
                        d_deltas_iter, delta_comp);

    const size_t n_new_leaves = d_tree.leaves.size();
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
    const AABBFunc AABB,
    const bool wipe = false)
{
    const TPrimitive* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    const DeltaType* deltas_ptr = thrust::raw_pointer_cast(d_deltas.data());

    build_ALBVH(d_tree, prims_ptr, deltas_ptr, delta_comp, AABB, wipe);
}

// Specialized with DeltaComp = thrust::less<DeltaType>
template <typename PrimitiveIter, typename DeltaIter, typename AABBFunc>
GRACE_HOST void build_ALBVH(
    Tree& d_tree,
    PrimitiveIter d_prims_iter,
    DeltaIter d_deltas_iter,
    const AABBFunc AABB,
    const bool wipe = false)
{
    typedef typename std::iterator_traits<DeltaIter>::value_type DeltaType;
    typedef typename thrust::less<DeltaType> DeltaComp;

    build_ALBVH(d_tree, d_prims_iter, d_deltas_iter, DeltaComp(), AABB, wipe);
}

// Specialized with DeltaComp = thrust::less<DeltaType>
template <typename TPrimitive, typename DeltaType, typename AABBFunc>
GRACE_HOST void build_ALBVH(
    Tree& d_tree,
    const thrust::device_vector<TPrimitive>& d_primitives,
    const thrust::device_vector<DeltaType>& d_deltas,
    const AABBFunc AABB,
    const bool wipe = false)
{
    typedef typename thrust::less<DeltaType> DeltaComp;
    const TPrimitive* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    const DeltaType* deltas_ptr = thrust::raw_pointer_cast(d_deltas.data());

    build_ALBVH(d_tree, prims_ptr, deltas_ptr, DeltaComp(), AABB, wipe);
}

} // namespace grace
