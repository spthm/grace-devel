#pragma once

// CUDA math constants.
#include <math_constants.h>
// CUDA math library.
#include <math.h>

#include <climits>
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

GRACE_HOST_DEVICE unsigned long long pack(int x, int y)
{
    typedef unsigned int       u32;
    typedef unsigned long long u64;

    GRACE_ASSERT(sizeof(int) * CHAR_BIT == 32);
    GRACE_ASSERT(sizeof(u32) * CHAR_BIT == 32);
    GRACE_ASSERT(sizeof(u64) * CHAR_BIT == 64);

    u32 u_x = reinterpret_cast<u32&>(x);
    u32 u_y = reinterpret_cast<u32&>(y);

    u64 ull_x = static_cast<u64>(u_x);
    u64 ull_y = static_cast<u64>(u_y);

    return (ull_x << 32) | ull_y;
}

GRACE_HOST_DEVICE int2 unpack(unsigned long long xy)
{
    typedef unsigned int       u32;
    typedef unsigned long long u64;

    u32 u_x = static_cast<u32>(xy >> 32);
    u32 u_y = static_cast<u32>(xy);

    int x = reinterpret_cast<int&>(u_x);
    int y = reinterpret_cast<int&>(u_y);

    return make_int2(x, y);
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

// value is the per-thread value for the reduction.
// lane is threadIdx.x % warpSize.
// shared is a pointer to a block of >= warpSize elements of type int.
// The pointer value should be identical for all threads in the warp, and no
// two warps should have overlapping ranges.
// On SM_30+ hardware, shared may have the size of a single int.
// After warp_reduction returns, the reduction is located in *shared, and is
// also returned to lane == 0.
GRACE_DEVICE int warp_reduce(int value, const int lane, volatile int* shared)
{
    GRACE_ASSERT(warpSize == grace::WARP_SIZE);

#if __CUDA_ARCH__ < 300

    shared[lane] = value;

    if (lane < grace::WARP_SIZE / 2) {
        #pragma unroll
        for (int i = grace::WARP_SIZE / 2; i > 0; i /= 2) {
            value += shared[lane + i]; shared[lane] = value;
        }
    }

#else

    #pragma unroll
    for (int i = grace::WARP_SIZE / 2; i > 0; i /= 2)
        value += __shfl_down(value, i);

    if (lane == 0) *shared = value;

#endif

    return value;
}

// value is the per-thread value for the reduction.
// lane is threadIdx.x % warpSize.
// wid is threadIdx.x / warpSize.
// shared is a pointer to a block of >= blockDim.x elements of type int.
// The pointer value should be identical for all threads in the block
// On SM_30+ hardware, shared may have the size of warpSize ints.
// After block_reduce returns, the reduction is located in *shared, and is
// also returned to lane == 0.
GRACE_DEVICE int block_reduce(int value, const int lane, const int wid,
                              const int n_warps, volatile int* shared)
{
    GRACE_ASSERT(n_warps <= grace::WARP_SIZE);

# if __CUDA_ARCH__ < 300

    value = warp_reduce(value, lane, shared + grace::WARP_SIZE * wid);
    __syncthreads();

    if (wid == 0) {
        // TODO: Bank conflicts here.
        value = (lane < n_warps) ? shared[grace::WARP_SIZE * lane] : 0;
        value = warp_reduce(value, lane, shared);
    }
    __syncthreads(); // so *shared == reduction for all threads.

#else

    value = warp_reduce(value, lane, shared + wid);
    __syncthreads();

    if (wid == 0) {
        value = (lane < n_warps) ? shared[lane] : 0;
        value = warp_reduce(value, lane, shared);
    }
    __syncthreads(); // so *shared == reduction for all threads.

#endif

    return value;
}

// value is the per-thread value for the scan.
// shared is a pointer to a block of >= 2 * warpSize ints.
// The pointer value should be identical for all threads in the warp.
// SM_30+ devices may provide a pointer to a single int for *shared.
// lane is threadIdx.x % warpSize.
// Returns to each thread the inclusive scan over values.
GRACE_DEVICE int warp_inc_scan(int value, const int lane, volatile int* shared)
{
    GRACE_ASSERT(grace::WARP_SIZE == warpSize);

#if __CUDA_ARCH__ < 300

    shared[lane] = 0;

    shared += grace::WARP_SIZE;
    shared[lane] = value;

    // ~ Hillis-Steele.
    #pragma unroll
    for (int i = grace::WARP_SIZE / 2; i > 0; i /= 2) {
        value += shared[lane - i]; shared[lane] = value;
    }

#else

    #pragma unroll
    for (int i = 1; i < grace::WARP_SIZE; i *= 2) {
        int red = __shfl_up(value, i);
        if (lane >= i) value += red;
    }

#endif

    return value;
}

GRACE_DEVICE int warp_ex_scan_pred(bool pred, const int lane, int* total)
{
    GRACE_ASSERT(warpSize == grace::WARP_SIZE);

    const unsigned int bitfield = __ballot(pred);
    *total = __popc(bitfield);

    const unsigned int ones = -1;
    const unsigned int lane_mask = ones >> (warpSize - lane);
    const int ex_scan = __popc(bitfield & lane_mask);

    return ex_scan;
}

// pred is the per-thread predicate to scan.
// lane is threadIdx.x % warpSize.
// wid is threadIdx.x / warpSize.
// shared is a pointer to a block of >= 2 * warpSize ints.
// SM_30+ devices may provide a pointer to only warpSize ints.
// The pointer value should be identical for all threads in the warp.
// lane is threadIdx.x % warpSize.
GRACE_DEVICE int block_ex_scan_pred(bool pred, const int lane, const int wid,
                                    const int n_warps, volatile int* shared)
{
    GRACE_ASSERT(n_warps <= grace::WARP_SIZE);

    int warp_total = 0;
    int lane_offset = warp_ex_scan_pred(pred, lane, &warp_total);

    if (lane == 0) shared[wid] = warp_total;
    __syncthreads();

    if (wid == 0) {
        int value = (lane < n_warps) ? shared[lane] : 0;
        shared[lane] = warp_inc_scan(value, lane, shared);
    }
    __syncthreads();

    // Exclusive scan.
    return shared[wid] - warp_total + lane_offset;
}

template <typename DeltaIter, typename DeltaComp>
__global__ void build_leaves_kernel(
    const int4* inq,
    const size_t n_in,
    DeltaIter deltas,
    const size_t n_primitives,
    const int max_per_leaf,
    const int climb_levels,
    int* node_ends,
    int4* outq,
    int* pool,
    int4* wide_leaves,
    int* spanned_primitives,
    const DeltaComp delta_comp)
{
    GRACE_ASSERT(grace::WARP_SIZE == warpSize);

    __shared__ int shared[grace::BUILD_THREADS_PER_BLOCK];
    __shared__ int block_offset;

    const int lane = threadIdx.x % grace::WARP_SIZE;
    const int wid = threadIdx.x / grace::WARP_SIZE;
    const int n_warps = grace::BUILD_THREADS_PER_BLOCK / grace::WARP_SIZE;


    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    ++deltas;

    int bid = blockIdx.x;
    for ( ; bid * grace::BUILD_THREADS_PER_BLOCK < n_in; bid += gridDim.x)
    {
        const int tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;
        const bool active = (tid < n_in);

        int4 node;
        int node_index;
        if (active) {
            node = inq[tid];
            node_index = node.x;
        }

        int total = 0;
        bool climb = active;
        for (int level = 0; level < climb_levels && climb; ++level)
        {
            GRACE_ASSERT(node.x >= 0 || node.x == node.z);
            GRACE_ASSERT(node.x < n_primitives - 1 || node.z == node.w);
            GRACE_ASSERT(node.z >= 0 || node.x == node.z);
            GRACE_ASSERT(node.z < n_primitives - 1 || node.z == node.w);
            GRACE_ASSERT(node.w > 0 || node.z == node.w);
            GRACE_ASSERT(node.w < n_primitives);

            if (node_size(node) > max_per_leaf)
            {
                const int2 lchild = make_int2(node.z, node_index);
                const int2 rchild = make_int2(node_index + 1, node.w);

                const int4 lleaf = make_int4(lchild.x, node_size(lchild),
                                             -1, -1);
                const int4 rleaf = make_int4(rchild.x, node_size(rchild),
                                             -1, -1);

                GRACE_ASSERT(wide_leaves[node_index].x     == -1 || lleaf.y > max_per_leaf);
                GRACE_ASSERT(wide_leaves[node_index + 1].x == -1 || rleaf.y > max_per_leaf);
                if (lleaf.y <= max_per_leaf) {
                    wide_leaves[node_index]     = lleaf;
                    total += lleaf.y;
                }
                if (rleaf.y <= max_per_leaf) {
                    wide_leaves[node_index + 1] = rleaf;
                    total += rleaf.y;
                }

                // We do not set climb to false, because our parent may still
                // need to be written to the out queue if our sibling is to
                // become a wide leaf.
            }

            // Root node has no valid parent index.
            if (node_size(node) < n_primitives)
            {
                const int parent_index = node_parent(node, deltas, delta_comp);
                GRACE_ASSERT(parent_index != node_index || node.z == node.w);

                const int this_end = is_left_child(node, parent_index) ? node.z : node.w;
                const int other_end = atomicExch(&node_ends[parent_index], this_end);
                GRACE_ASSERT(other_end != this_end);

                // Second thread to reach the parent is the one which continues.
                climb = (other_end != -1);
                node = make_int4(parent_index, -1,
                                 min(this_end, other_end),
                                 max(this_end, other_end));
                node_index = parent_index;
            }
            else {
                // Can't climb further than the root.
                // Can't simply return; need to reach the __syncthreads() below
                // to avoid deadlock.
                climb = false;
            }
        } // for (level < climb_levels && climb)

        // Any thread wishing to continue the climb beyond climb_levels should
        // output its parent to the queue.
        const bool join = climb;
        const int4 join_node = node;

        // Make sure we can safely use shared.
        __syncthreads();

        // Per-block update of spanned_primitives.
        // total = block_reduce(total, lane, wid, n_warps, shared);
        // if (threadIdx.x == 0 && total) atomicAdd(spanned_primitives, total);

        // Per-warp update of spanned_primitives.
        if (__ballot(total != 0)) {
            total = warp_reduce(total, lane, shared + wid * grace::WARP_SIZE);
        }
        if (lane == 0 && total) atomicAdd(spanned_primitives, total);

        // Per-thread update of spanned_primitives.
        // On Kepler, this seems marginally faster.
        // if (total) atomicAdd(spanned_primitives, total);

        // Per-block update of work-queue pool.
        // On Kepler, this is fastest.
        // Make sure we're done with shared before we use it again.
        total = __syncthreads_count(join);
        const int thread_offset = block_ex_scan_pred(join, lane, wid, n_warps,
                                                     shared);
        if (threadIdx.x == 0) block_offset = atomicAdd(pool, total);
        __syncthreads();
        if (join) outq[block_offset + thread_offset] = join_node;

        // Per-warp update of work-queue pool.
        // int lane_offset = warp_ex_scan_pred(join, lane, &total);
        // if (lane == 0) shared[wid] = atomicAdd(pool, total);
        // if (join) outq[shared[wid] + lane_offset] = join_node;

        // Per-thread update of work-queue pool.
        // int offset = atomicAdd(pool, (int)join);
        // if (join) outq[offset] = join_node;
    } // for bid * BUILD_THREADS_PER_BLOCK < n_in
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
__global__ void join_leaves_kernel(
    int4* nodes,
    const int4* leaves,
    const size_t n_leaves,
    const int iteration,
    PrimitiveIter primitives,
    DeltaIter deltas,
    unsigned long long* flags,
    int* outq,
    int* pool,
    const DeltaComp delta_comp,
    const AABBFunc AABB)
{
    GRACE_ASSERT(grace::WARP_SIZE == warpSize);

    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;
    __shared__ int shared[grace::BUILD_THREADS_PER_BLOCK];
    __shared__ int block_offset;

    const int lane = threadIdx.x % grace::WARP_SIZE;
    const int wid = threadIdx.x / grace::WARP_SIZE;
    const int n_warps = grace::BUILD_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const int n_nodes = n_leaves - 1;

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    ++deltas;

    int bid = blockIdx.x;
    for ( ; bid * grace::BUILD_THREADS_PER_BLOCK < n_leaves; bid += gridDim.x)
    {
        const int tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;

        int parent_index;
        bool join = false;
        if (tid < n_leaves)
        {
            int4 leaf = get_leaf(tid, leaves);

            GRACE_ASSERT(leaf.x >= 0);
            GRACE_ASSERT(leaf.y > 0);
            GRACE_ASSERT(leaf.y < n_leaves);
            GRACE_ASSERT(leaf.z == tid);
            GRACE_ASSERT(leaf.w == tid);

            parent_index = node_parent(leaf, deltas, delta_comp);

            float3 bot, top;
            bot.x = bot.y = bot.z = CUDART_INF_F;
            top.x = top.y = top.z = -1.f;

            #pragma unroll 4
            for (int i = 0; i < leaf.y; i++) {
                TPrimitive prim = primitives[leaf.x + i];

                float3 pbot, ptop;
                AABB(prim, &pbot, &ptop);

                AABB_union(bot, top, pbot, ptop, &bot, &top);
            }

            // Note, they should never be equal.
            GRACE_ASSERT(bot.x < top.x);
            GRACE_ASSERT(bot.y < top.y);
            GRACE_ASSERT(bot.z < top.z);

            if (is_left_child(leaf, parent_index)) {
                // Leaf indices are offset by n_nodes.
                propagate_left(leaf, bot, top, tid + n_nodes, parent_index,
                               nodes);
            }
            else {
                // Leaf indices are offset by n_nodes.
                propagate_right(leaf, bot, top, tid + n_nodes, parent_index,
                                nodes);
            }

            // The second thread to reach a (parent) node will write that node's
            // index to the work queue.
            const unsigned long long thread_code = pack(iteration, tid);
            const unsigned long long other_code = atomicExch(flags + parent_index, thread_code);
            const int2 other = unpack(other_code);
            const int other_idx = other.y;
            GRACE_ASSERT(other_idx != tid);
            join = (other_idx != -1);
        } // if (tid < n_leaves);

        // Per-block update of work-queue pool.
        // On Kepler, this is fastest.
        int total = __syncthreads_count(join);
        const int thread_offset = block_ex_scan_pred(join, lane, wid, n_warps,
                                                     shared);

        if (threadIdx.x == 0) block_offset = atomicAdd(pool, total);
        __syncthreads();

        if (join) {
            outq[block_offset + thread_offset] = parent_index;
        }

        // Per-warp update of work-queue pool.
        // int total = 0;
        // int lane_offset = warp_ex_scan_pred(join, lane, &total);
        // if (lane == 0 && total) shared[wid] = atomicAdd(pool, total);
        // if (join) outq[shared[wid] + lane_offset] = parent_index;

    } // for bid * BUILD_THREADS_PER_BLOCK < n_leaves
}

template <
    typename PrimitiveIter,
    typename DeltaIter,
    typename DeltaComp,
    typename AABBFunc
    >
__global__ void join_nodes_kernel(
    const int* inq,
    const size_t n_in,
    const int iteration,
    int4* nodes,
    const size_t n_nodes,
    int* root_index,
    PrimitiveIter primitives,
    DeltaIter deltas,
    const int climb_levels,
    unsigned long long int* flags,
    int* outq,
    int* pool,
    const DeltaComp delta_comp,
    const AABBFunc AABB)
{
    GRACE_ASSERT(grace::WARP_SIZE == warpSize);

    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;

    union Shared {
        int scan[grace::BUILD_THREADS_PER_BLOCK];
        int other_end[grace::BUILD_THREADS_PER_BLOCK];
    };

    __shared__ float3 AABB_min[grace::BUILD_THREADS_PER_BLOCK];
    __shared__ float3 AABB_max[grace::BUILD_THREADS_PER_BLOCK];
    __shared__ Shared shared;
    __shared__ int block_offset;

    const int lane = threadIdx.x % grace::WARP_SIZE;
    const int wid = threadIdx.x / grace::WARP_SIZE;
    const int n_warps = grace::BUILD_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t n_leaves = n_nodes + 1;

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    ++deltas;

    int bid = blockIdx.x;
    for ( ; bid * grace::BUILD_THREADS_PER_BLOCK < n_in; bid += gridDim.x)
    {
        const int tid = threadIdx.x + bid * grace::BUILD_THREADS_PER_BLOCK;
        const int low = bid * grace::BUILD_THREADS_PER_BLOCK;
        const int high = (bid + 1) * grace::BUILD_THREADS_PER_BLOCK;
        const bool active = (tid < n_in);

        int node_index, other_idx, other_iteration;
        int4 node;
        float3 bot, top;
        if (active) {
            node_index = inq[tid];
            node = get_inner(node_index, nodes);

            // Set up AABB_min[other_idx], AABB_max[other_idx] and
            // shared.other_end[other_idx] as if we have come from the right
            // child of node.
            other_idx = tid;
            other_iteration = iteration;
            shared.other_end[threadIdx.x] = node.z;

            get_left_AABB(node_index, nodes, &bot, &top);
            AABB_min[threadIdx.x] = bot;
            AABB_max[threadIdx.x] = top;

            get_right_AABB(node_index, nodes, &bot, &top);
        }

        bool climb = active;
        for (int level = 0; level < climb_levels && climb; ++level)
        {
            GRACE_ASSERT(node_index >= 0 && node_index < n_nodes);

            const bool from_left = is_left_child(node, node_index);

            float3 other_bot, other_top;
            if (other_iteration != iteration ||
                out_of_block(other_idx, low, high))
            {
                node = get_inner(node_index, nodes);
                if (from_left) {
                    get_right_AABB(node_index, nodes, &other_bot, &other_top);
                }
                else {
                    get_left_AABB(node_index, nodes, &other_bot, &other_top);
                }
            }
            else
            {
                const int other_end = shared.other_end[other_idx - low];
                if (from_left) node.w = other_end;
                else           node.z = other_end;

                other_bot = AABB_min[other_idx - low];
                other_top = AABB_max[other_idx - low];
            }

            AABB_union(bot, top, other_bot, other_top, &bot, &top);


            GRACE_ASSERT(node.x >= 0 && node.x < n_nodes + n_leaves - 1);
            GRACE_ASSERT(node.y > 0 && node.w <= n_nodes + n_leaves - 1);
            GRACE_ASSERT(node.z >= 0 && node.z < n_leaves - 1);
            GRACE_ASSERT(node.w > 0 && node.w <= n_leaves - 1);

            // Note, they should never be equal.
            GRACE_ASSERT(bot.x < top.x);
            GRACE_ASSERT(bot.y < top.y);
            GRACE_ASSERT(bot.z < top.z);

            AABB_min[threadIdx.x] = bot;
            AABB_max[threadIdx.x] = top;

            // Root node has no valid parent index.
            if (node_size(node) < n_leaves)
            {
                const int parent_index = node_parent(node, deltas, delta_comp);

                if (is_left_child(node, parent_index)) {
                    propagate_left(node, bot, top, node_index, parent_index,
                                   nodes);
                    shared.other_end[threadIdx.x] = node.z;
                }
                else {
                    propagate_right(node, bot, top, node_index, parent_index,
                                    nodes);
                    shared.other_end[threadIdx.x] = node.w;
                }

                // If another thread sees other_idx != -1 for the current
                // parent, it must also be able to see our writes to the
                // current parent.
                __threadfence();

                const unsigned long long thread_code = pack(iteration, tid);
                const unsigned long long other_code = atomicExch(flags + parent_index, thread_code);
                GRACE_ASSERT(other_code != thread_code);
                const int2 other = unpack(other_code);
                other_iteration = other.x;
                other_idx = other.y;
                // Second thread to reach the parent is the one which continues.
                climb = (other_idx != -1);

                node_index = parent_index;
            }
            else {
                // Can't climb further than the root.
                // Can't simply return; need to reach the __syncthreads() below
                // to avoid deadlock.
                *root_index = node_index;
                climb = false;
            }
        } // for (level < climb_levels && climb);

        // Any thread wishing to climb beyond climb_levels should output its
        // current parent index to the queue.
        const bool join = climb;
        const int join_node_index = node_index;

        // Per-block update of work-queue pool.
        // On Kepler, this is fastest.
        int total = __syncthreads_count(join);
        const int thread_offset = block_ex_scan_pred(join, lane, wid, n_warps,
                                                     shared.scan);

        if (threadIdx.x == 0) block_offset = atomicAdd(pool, total);
        __syncthreads();

        if (join) {
            outq[block_offset + thread_offset] = join_node_index;
        }

        // Per-warp update of work-queue pool.
        // int total = 0;
        // int lane_offset = warp_ex_scan_pred(join, lane, &total);
        // if (lane == 0 && total) shared.scan[wid] = atomicAdd(pool, total);
        // if (join) outq[shared.scan[wid] + lane_offset] = parent_index;
    } // for bid * BUILD_THREADS_PER_BLOCK < n_in
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

struct int4_functor
{
    GRACE_HOST_DEVICE int4 operator()(unsigned int n)
    {
        return make_int4(n, n, n, n);
    }
};

template <typename DeltaIter, typename DeltaComp>
GRACE_HOST void build_leaves(
    Tree& d_tree,
    DeltaIter d_deltas_iter,
    const DeltaComp delta_comp)
{
    const size_t n_primitives = d_tree.leaves.size();
    const int max_per_leaf = d_tree.max_per_leaf;

    if (n_primitives <= max_per_leaf) {
        const std::string msg
            = "max_per_leaf must be less than the total number of primitives.";
        throw std::invalid_argument(msg);
    }

    thrust::device_vector<int> d_node_ends(n_primitives - 1);
    thrust::device_vector<int4> d_queue1(n_primitives);
    thrust::device_vector<int4> d_queue2(n_primitives / 2);
    thrust::device_vector<int> d_pool(1);
    thrust::device_vector<int> d_completed(1);

    thrust::fill(d_tree.leaves.begin(), d_tree.leaves.end(),
                 invalid<int4>::node);
    thrust::fill(d_node_ends.begin(), d_node_ends.end(), -1);
    thrust::tabulate(d_queue1.begin(), d_queue1.end(), int4_functor());
    d_completed[0] = 0;

    int4* d_in_queue  = thrust::raw_pointer_cast(d_queue1.data());
    int4* d_out_queue = thrust::raw_pointer_cast(d_queue2.data());

    size_t n_in = n_primitives;
    int n_completed = 0;
    while (n_completed < n_primitives)
    {
        d_pool[0] = 0u;

        int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_in + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));
        build_leaves_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
            d_in_queue,
            n_in,
            d_deltas_iter,
            n_primitives,
            max_per_leaf,
            10, // On Kepler, this is optimal for max_per_leaf = 32, 64 and 128.
            thrust::raw_pointer_cast(d_node_ends.data()),
            d_out_queue,
            thrust::raw_pointer_cast(d_pool.data()),
            thrust::raw_pointer_cast(d_tree.leaves.data()),
            thrust::raw_pointer_cast(d_completed.data()),
            delta_comp);

        n_in = d_pool[0];
        n_completed = d_completed[0];

        std::swap(d_in_queue, d_out_queue);
    }

    // Using
    //   const invalid4 = make_int4(-1, -1, -1, -1);
    //   thrust::remove(..., invalid4);
    // gives:
    //   ... error: no operator "==" matches these operands
    //           operand types are: const int4 == const int4
    d_tree.leaves.erase(
        thrust::remove_if(d_tree.leaves.begin(), d_tree.leaves.end(),
                          is_invalid_node<int4>()),
        d_tree.leaves.end());
    const size_t n_wide_leaves = d_tree.leaves.size();
    d_tree.nodes.resize(4 * n_wide_leaves);

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_wide_leaves + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));

    fix_leaf_ranges<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        n_wide_leaves
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

    thrust::device_vector<unsigned long long int> d_flags(n_nodes);
    thrust::device_vector<int> d_queue1(n_leaves / 2);
    thrust::device_vector<int> d_queue2(n_leaves / 2);
    thrust::device_vector<int> d_pool(1);

    const unsigned long long flag = pack(0, -1);
    thrust::fill(d_flags.begin(), d_flags.end(), flag);
    d_pool[0] = 0;
    int iteration = 0;

    int* d_in_queue = thrust::raw_pointer_cast(d_queue1.data());
    int* d_out_queue = thrust::raw_pointer_cast(d_queue2.data());

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_leaves + grace::BUILD_THREADS_PER_BLOCK - 1)
                            / grace::BUILD_THREADS_PER_BLOCK));
    join_leaves_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tree.nodes.data()),
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        n_leaves,
        iteration,
        d_prims_iter,
        d_deltas_iter,
        thrust::raw_pointer_cast(d_flags.data()),
        d_in_queue, // Will be used as input for first iteration of while().
        thrust::raw_pointer_cast(d_pool.data()),
        delta_comp,
        AABB);

    size_t n_in = d_pool[0];
    // join_leaves_kernel was iteration 1.
    ++iteration;
    while (n_in)
    {
        d_pool[0] = 0;

        blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_in + grace::BUILD_THREADS_PER_BLOCK - 1)
                             / grace::BUILD_THREADS_PER_BLOCK));

        join_nodes_kernel<<<blocks, grace::BUILD_THREADS_PER_BLOCK>>>(
            d_in_queue,
            n_in,
            iteration,
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            n_nodes,
            d_tree.root_index_ptr,
            d_prims_iter,
            d_deltas_iter,
            10, // TODO: Choose optimal value, or find suitable heuristic.
            thrust::raw_pointer_cast(d_flags.data()),
            d_out_queue,
            thrust::raw_pointer_cast(d_pool.data()),
            delta_comp,
            AABB);
        GRACE_KERNEL_CHECK();

        n_in = d_pool[0];
        ++iteration;
        std::swap(d_in_queue, d_out_queue);
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

    ALBVH::build_leaves(d_tree, d_deltas_iter, delta_comp);

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
