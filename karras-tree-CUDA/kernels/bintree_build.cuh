#pragma once

#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

// CUDA math constants.
#include <math_constants.h>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/swap.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

//-----------------------------------------------------------------------------
// Helper function for tree build kernels.
//-----------------------------------------------------------------------------

struct is_empty_node : public thrust::unary_function<int4, bool>
{
    __host__ __device__
    bool operator()(const int4 node) const
    {
        // Note: a node's right child can never be node 0, and a leaf can never
        // cover zero elements.
        return (node.y == 0);
    }
};

namespace gpu {

//-----------------------------------------------------------------------------
// GPU helper functions for tree build kernels.
//-----------------------------------------------------------------------------

__device__ uinteger32 node_delta(const int i,
                                 const uinteger32* keys,
                                 const size_t n_keys)
{
    // delta(-1) and delta(N-1) must return e.g. UINT_MAX because they cover
    // invalid ranges but are valid queries during tree construction.
    if (i < 0 || i + 1 >= n_keys)
        return uinteger32(-1);

    uinteger32 ki = keys[i];
    uinteger32 kj = keys[i+1];

    return ki ^ kj;

}

__device__ uinteger64 node_delta(const int i,
                                 const uinteger64* keys,
                                 const size_t n_keys)
{
    // delta(-1) and delta(N-1) must return e.g. UINT_MAX because they cover
    // invalid ranges but are valid queries during tree construction.
    if (i < 0 || i + 1 >= n_keys)
        return uinteger64(-1);

    uinteger64 ki = keys[i];
    uinteger64 kj = keys[i+1];

    return ki ^ kj;

}

// Euclidian distance metric.
template <typename Float4>
__device__ float node_delta(const int i,
                            const Float4* spheres,
                            const size_t n_spheres)
{
    if (i < 0 || i + 1 >= n_spheres)
        return CUDART_INF_F;

    Float4 si = spheres[i];
    Float4 sj = spheres[i+1];

    return (si.x - sj.x) * (si.x - sj.x)
           + (si.y - sj.y) * (si.y - sj.y)
           + (si.z - sj.z) * (si.z - sj.z);
}

// Surface area 'distance' metric.
// template <typename Float4>
// __device__ float node_delta(const int i,
//                             const Float4* spheres,
//                             const size_t n_spheres)
// {
//     if (i < 0 || i + 1 >= n_spheres)
//         return CUDART_INF_F;

//     Float4 si = spheres[i];
//     Float4 sj = spheres[i+1];

//     float L_x = max(si.x + si.w, sj.x + sj.w) - min(si.x - si.w, sj.x - sj.w);
//     float L_y = max(si.y + si.w, sj.y + sj.w) - min(si.y - si.w, sj.y - sj.w);
//     float L_z = max(si.z + si.w, sj.z + sj.w) - min(si.z - si.w, sj.z - sj.w);

//     float SA = (L_x * L_y) + (L_x * L_z) + (L_y * L_z);

//     assert(SA < CUDART_INF_F);
//     assert(SA > 0);

//     return SA;
// }

// Load/store functions to be used when specific memory behaviour is required,
// e.g. a read/write directly from/to L2 cache, which is globally coherent.
//
// All take a pointer with the type of the base primitive (i.e. int* for int2
// read/writes, float* for float4 read/writes) for flexibility.
//
// The "memory" clobber is added to all load/store PTX instructions to prevent
// memory optimizations around the asm statements. We should only be using these
// functions when we know better than the compiler!
__device__ __forceinline__ int2 load_vec2s32(const int* const addr)
{
    int2 i2;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.ca.v2.s32 {%0, %1}, [%2];" : "=r"(i2.x), "=r"(i2.y) : "l"(addr) : "memory");
    #else
    asm("ld.global.ca.v2.s32 {%0, %1}, [%2];" : "=r"(i2.x), "=r"(i2.y) : "r"(addr) : "memory");
    #endif

    return i2;
}

__device__ __forceinline__ int4 load_vec4s32(const int* const addr)
{
    int4 i4;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.ca.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(i4.x), "=r"(i4.y), "=r"(i4.z), "=r"(i4.w) : "l"(addr) : "memory");
    #else
    asm("ld.global.ca.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(i4.x), "=r"(i4.y), "=r"(i4.z), "=r"(i4.w) : "r"(addr) : "memory");
    #endif

    return i4;
}

__device__ __forceinline__ float4 load_vec4f32(const float* const addr)
{
    float4 f4;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "l"(addr) : "memory");
    #else
    asm("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "r"(addr) : "memory");
    #endif

    return f4;
}

__device__ __forceinline__ float4 load_L2_vec4f32(const float* const addr)
{
    float4 f4;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "l"(addr) : "memory");
    #else
    asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "r"(addr) : "memory");
    #endif

    return f4;
}

// Stores have no output operands, so we additionally mark them as volatile to
// ensure they are not moved or deleted.
__device__ __forceinline__ void store_s32(const int* const addr, const int a)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.s32 [%0], %1;" :: "l"(addr), "r"(a) : "memory");
    #else
    asm volatile ("st.global.wb.s32 [%0], %1;" :: "r"(addr), "r"(a) : "memory");
    #endif
}

__device__ __forceinline__ void store_vec2s32(
    const int* const addr,
    const int a, const int b)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.v2.s32 [%0], {%1, %2};" :: "l"(addr), "r"(a), "r"(b) : "memory");
    #else
    asm volatile ("st.global.wb.v2.s32 [%0], {%1, %2};" :: "r"(addr), "r"(a), "r"(b) : "memory");
    #endif
}

__device__ __forceinline__ void store_vec2f32(
    const float* const addr,
    const float a, const float b)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.v2.f32 [%0], {%1, %2};" :: "l"(addr), "f"(a), "f"(b) : "memory");
    #else
    asm volatile ("st.global.wb.v2.f32 [%0], {%1, %2};" :: "r"(addr), "f"(a), "f"(b) : "memory");
    #endif
}

__device__ __forceinline__ void store_vec4f32(
    const float* const addr,
    const float a, const float b, const float c, const float d)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #else
    asm volatile ("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #endif
}

__device__ __forceinline__ void store_L2_vec2f32(
    const float* const addr,
    const float a, const float b)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.cg.v2.f32 [%0], {%1, %2};" :: "l"(addr), "f"(a), "f"(b) : "memory");
    #else
    asm volatile ("st.global.cg.v2.f32 [%0], {%1, %2};" :: "r"(addr), "f"(a), "f"(b) : "memory");
    #endif
}

__device__ __forceinline__ void store_L2_vec4f32(
    const float* const addr,
    const float a, const float b, const float c, const float d)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #else
    asm volatile ("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #endif
}

//-----------------------------------------------------------------------------
// CUDA kernels for tree building.
//-----------------------------------------------------------------------------

template <typename KeyType, typename DeltaType>
__global__ void compute_deltas_kernel(const KeyType* keys,
                                      const size_t n_keys,
                                      DeltaType* deltas)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid <= n_keys)
    {
        // The range [-1, n_keys) is valid for querying node_delta.
        deltas[tid] = node_delta(tid - 1, keys, n_keys);
        tid += blockDim.x * gridDim.x;
    }
}

template <typename KeyType, typename DeltaType>
__global__ void compute_leaf_deltas_kernel(const int4* leaves,
                                           const size_t n_leaves,
                                           const KeyType* keys,
                                           const size_t n_keys,
                                           DeltaType* deltas)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // The range [-1, n_leaves) is valid for querying node_delta.
    if (tid == 0)
        deltas[0] = node_delta(-1, keys, n_keys);

    while (tid < n_leaves)
    {
        int4 leaf = leaves[tid];
        int last_idx = leaf.x + leaf.y - 1;
        deltas[tid+1] = node_delta(last_idx, keys, n_keys);
        tid += blockDim.x * gridDim.x;
    }
}

template <typename DeltaType>
__global__ void build_leaves_kernel(int2* nodes,
                                    const size_t n_nodes,
                                    const DeltaType* deltas,
                                    const int max_per_leaf)
{
    extern __shared__ int flags[];

    const size_t n_leaves = n_nodes + 1;

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    deltas++;

    int bid = blockIdx.x;
    int tid = threadIdx.x + bid * BUILD_THREADS_PER_BLOCK;
    // while() and an inner for() to ensure all threads in a block hit the
    // __syncthreads() and wipe the flags.
    while (bid * BUILD_THREADS_PER_BLOCK < n_leaves)
    {
        // Zero all SMEM flags at start of first loop and at end of subsequent
        // loops.
        __syncthreads();
        for (int i = threadIdx.x;
             i < BUILD_THREADS_PER_BLOCK + max_per_leaf;
             i += BUILD_THREADS_PER_BLOCK)
        {
            flags[i] = 0;
        }
        __syncthreads();

        // [low, high) leaf indices covered by this block, including the
        // max_per_leaf buffer.
        int low = bid * BUILD_THREADS_PER_BLOCK;
        int high = min((bid + 1) * BUILD_THREADS_PER_BLOCK + max_per_leaf,
                       (int)n_leaves);

        for (int idx = tid; idx < high; idx += BUILD_THREADS_PER_BLOCK)
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
            if (delta_L < delta_R)
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
            store_s32(addr, end);

            // Travel up the tree.  The second thread to reach a node writes its
            // left or right end to its parent.  The first exits the loop.
            __threadfence_block();
            assert(parent_index - low >= 0);
            assert(parent_index - low < BUILD_THREADS_PER_BLOCK + max_per_leaf);
            unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
            assert(flag < 2);
            bool first_arrival = (flag == 0);

            while (!first_arrival)
            {
                cur_index = parent_index;

                // We are certain that a thread in this block has already
                // written the other child of the current node, so we can read
                // from L1 cache if we get a hit.  Normal vector int2 load.
                int2 left_right = load_vec2s32(&(nodes[parent_index].x));
                int left = left_right.x;
                int right = left_right.y;

                // Only the left-most leaf can have an index of 0, and only the
                // right-most leaf can have an index of n_leaves - 1.
                assert(left >= 0);
                assert(left < n_leaves - 1);
                assert(right > 0);
                assert(right < n_leaves);

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
                if (delta_L < delta_R)
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
                store_s32(addr, end);

                __threadfence_block();
                assert(parent_index - low >= 0);
                assert(parent_index - low < BUILD_THREADS_PER_BLOCK + max_per_leaf);
                unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
                assert(flag < 2);
                first_arrival = (flag == 0);
            } // while (!first_arrival)
        } // for idx = [threadIdx.x, BUILD_THREADS_PER_BLOCK + max_per_leaf)
        bid += gridDim.x;
        tid = threadIdx.x + bid * BUILD_THREADS_PER_BLOCK;
    } // while (bid * BUILD_THREADS_PER_BLOCK < n_leaves)
    return;
}

__global__ void write_leaves_kernel(const int2* nodes,
                                    const size_t n_nodes,
                                    int4* big_leaves,
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
        int4 leaf;
        if (left_leaf && write_check) {
            leaf.x = left;
            leaf.y = left_size;
            big_leaves[left] = leaf;
        }
        if (right_leaf && write_check) {
            leaf.x = tid + 1;
            leaf.y = right_size;
            big_leaves[right] = leaf;
        }

        tid += blockDim.x * gridDim.x;
    }
}

template <typename DeltaType, typename Float4>
__global__ void build_nodes_slice_kernel(int4* nodes,
                                         float4* f4_nodes,
                                         const size_t n_nodes,
                                         const int4* leaves,
                                         const size_t n_leaves,
                                         const Float4* spheres,
                                         const int* base_indices,
                                         const size_t n_base_nodes,
                                         int* root_index,
                                         const DeltaType* deltas,
                                         const int max_per_node,
                                         int* new_base_indices)
{
    extern __shared__ int SMEM[];

    int* flags = SMEM;
    volatile int2* sm_nodes
        = (int2*)&flags[BUILD_THREADS_PER_BLOCK + max_per_node];

    // Offset deltas so the range [-1, n_leaves) is valid for indexing it.
    deltas++;

    int bid = blockIdx.x;
    int tid = threadIdx.x + bid * BUILD_THREADS_PER_BLOCK;

    // while() and inner for() loops to ensure all threads in a block hit the
    // __syncthreads().
    while (bid * BUILD_THREADS_PER_BLOCK < n_base_nodes)
    {
        // Zero all SMEM flags and node data at start of first loop and at end
        // of subsequent loops.
        __syncthreads();
        for (int i = threadIdx.x;
             i < BUILD_THREADS_PER_BLOCK + max_per_node;
             i += BUILD_THREADS_PER_BLOCK)
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
        int low = bid * BUILD_THREADS_PER_BLOCK;
        int high = min((bid + 1) * BUILD_THREADS_PER_BLOCK + max_per_node,
                       (int)n_base_nodes);

        for (int idx = tid; idx < high; idx += BUILD_THREADS_PER_BLOCK)
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
            assert(g_cur_index < n_nodes + n_leaves);

            int4 node;
            if (g_cur_index < n_nodes)
                node = nodes[4 * g_cur_index + 0];
            else
                node = leaves[g_cur_index - n_nodes];

            float x_min, y_min, z_min;
            float x_max, y_max, z_max;
            if (g_cur_index < n_nodes) {
                float4 AABB_L  = f4_nodes[4 * g_cur_index + 1];
                float4 AABB_R  = f4_nodes[4 * g_cur_index + 2];
                float4 AABB_LR = f4_nodes[4 * g_cur_index + 3];

                // Compute the current node's AABB (union of its children's
                // AABBs).
                x_min = min(AABB_L.x, AABB_R.x);
                x_max = max(AABB_L.y, AABB_R.y);

                y_min = min(AABB_L.z, AABB_R.z);
                y_max = max(AABB_L.w, AABB_R.w);

                z_min = min(AABB_LR.x, AABB_LR.z);
                z_max = max(AABB_LR.y, AABB_LR.w);
            }
            else {
                x_min = y_min = z_min = CUDART_INF_F;
                x_max = y_max = z_max = -1.f;

                // Current node is a leaf; compute it's AABB from the spheres
                // it contains.
                #pragma unroll 4
                for (int i = 0; i < node.y; i++) {
                    Float4 sphere = spheres[node.x + i];

                    x_min = min(x_min, sphere.x - sphere.w);
                    x_max = max(x_max, sphere.x + sphere.w);

                    y_min = min(y_min, sphere.y - sphere.w);
                    y_max = max(y_max, sphere.y + sphere.w);

                    z_min = min(z_min, sphere.z - sphere.w);
                    z_max = max(z_max, sphere.z + sphere.w);
                }
            }

            // Note, they should never be equal.
            assert(x_min < x_max);
            assert(y_min < y_max);
            assert(z_min < z_max);

            // Recall that leaf left/right limits are equal to the leaf index,
            // minus the leaf-identifying offset.
            int g_left = (g_cur_index < n_nodes ? node.z :
                                                  g_cur_index - n_nodes);
            int g_right = (g_cur_index < n_nodes ? node.w :
                                                   g_cur_index - n_nodes);

            // Only the left-most leaf can have an index of 0, and only the
            // right-most leaf can have an index of n_leaves - 1.
            if (g_cur_index < n_nodes) {
                assert(g_left >= 0);
                assert(g_left < n_leaves - 1);
                assert(g_right > 0);
                assert(g_right < n_leaves);
            }
            else {
                assert(g_left >= 0);
                assert(g_left < n_leaves);
                assert(g_right >= 0);
                assert(g_right < n_leaves);
            }

            int left = cur_index;
            int right = cur_index;

            DeltaType delta_L = deltas[g_left - 1];
            DeltaType delta_R = deltas[g_right];
            int* child_addr;
            int* end_addr;
            volatile int* sm_addr;
            float* AABB_f2_addr;
            float* AABB_f4_addr;
            int end, sm_end;
            // Compute the current node's parent index and write-to locations.
            if (delta_L < delta_R)
            {
                // Leftward node is parent.
                parent_index = left - 1;
                g_parent_index = g_left - 1;

                // Current node is a right child.
                // The -ve end index encodes the fact that the node was written
                // this iteration.
                child_addr = &(nodes[4 * g_parent_index + 0].y);
                end_addr = &(nodes[4 * g_parent_index + 0].w);
                end = -1 * (right + 1);

                // Right-most global node index.
                sm_addr = &(sm_nodes[parent_index - low].y);
                sm_end = g_right;

                // Right child AABB.
                AABB_f4_addr = &(f4_nodes[4 * g_parent_index + 2].x);
                AABB_f2_addr = &(f4_nodes[4 * g_parent_index + 3].z);
            }
            else {
                // Rightward node is parent.
                parent_index = right;
                g_parent_index = g_right;

                // Current node is a left child.
                // The -ve end index encodes the fact that the node was written
                // this iteration.
                child_addr = &(nodes[4 * g_parent_index + 0].x);
                end_addr = &(nodes[4 * g_parent_index + 0].z);
                end = -1 * (left + 1);

                // Left-most global node index.
                sm_addr = &(sm_nodes[parent_index - low].x);
                sm_end = g_left;

                // Left child AABB.
                AABB_f4_addr = &(f4_nodes[4 * g_parent_index + 1].x);
                AABB_f2_addr = &(f4_nodes[4 * g_parent_index + 3].x);
            }

            // If the leaf's parent is outside this block do not write anything;
            // an adjacent block will follow this path.
            if (parent_index < low || parent_index >= high)
                continue;

            // Parent index must be at a valid location for writing to sm_nodes
            // and the SMEM flags.
            assert(parent_index - low >= 0);
            assert(parent_index - low < BUILD_THREADS_PER_BLOCK + max_per_node);

            // Normal stores.  Other threads in this block can read from L1 if
            // they get a cache hit, i.e. there is no requirement for global
            // coherency.
            store_s32(child_addr, g_cur_index);
            store_s32(end_addr, end);
            store_vec4f32(AABB_f4_addr, x_min, x_max, y_min, y_max);
            store_vec2f32(AABB_f2_addr, z_min, z_max);
            *sm_addr = sm_end;

            // Travel up the tree.  The second thread to reach a node writes its
            // logical/compressed left or right end to its parent; the first
            // exits the loop.
            __threadfence_block();
            unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
            assert(flag < 2);
            bool first_arrival = (flag == 0);

            while (!first_arrival)
            {
                cur_index = parent_index;
                g_cur_index = g_parent_index;

                assert(cur_index < n_base_nodes);
                assert(g_cur_index < n_nodes);

                // We are certain that a thread in this block has already
                // *written* the other child of the current node, so we can read
                // from L1 if we get a cache hit.
                int4 node = load_vec4s32(&(nodes[4 * g_cur_index + 0].x));

                // 'error: class "int2" has no suitable copy constructor'
                // int2 left_right = sm_nodes[cur_index - low];
                // int g_left = left_right.x;
                // int g_right = left_right.y;
                int g_left = sm_nodes[cur_index - low].x;
                int g_right = sm_nodes[cur_index - low].y;
                assert(g_left >= 0);
                assert(g_left < n_leaves - 1);
                assert(g_right > 0);
                assert(g_right < n_leaves);

                // Undo the -ve sign encoding.
                int left = -1 * node.z - 1;
                int right = -1 * node.w - 1;
                // We are the second thread in this block to reach this node.
                // Both of its logical/compressed end-indices must also be in
                // this block.
                assert(left >= low);
                assert(left < high - 1);
                assert(right > low);
                assert(right < high);

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
                float4 AABB_L  = load_vec4f32(&(f4_nodes[4 * g_cur_index + 1].x));
                float4 AABB_R  = load_vec4f32(&(f4_nodes[4 * g_cur_index + 2].x));
                float4 AABB_LR = load_vec4f32(&(f4_nodes[4 * g_cur_index + 3].x));

                float x_min = min(AABB_L.x, AABB_R.x);
                float x_max = max(AABB_L.y, AABB_R.y);
                float y_min = min(AABB_L.z, AABB_R.z);
                float y_max = max(AABB_L.w, AABB_R.w);
                float z_min = min(AABB_LR.x, AABB_LR.z);
                float z_max = max(AABB_LR.y, AABB_LR.w);

                assert(x_min < x_max);
                assert(y_min < y_max);
                assert(z_min < z_max);

                DeltaType delta_L = deltas[g_left - 1];
                DeltaType delta_R = deltas[g_right];
                int* child_addr;
                int* end_addr;
                volatile int* sm_addr;
                float* AABB_f2_addr;
                float* AABB_f4_addr;
                int end, sm_end;
                if (delta_L < delta_R)
                {
                    // Leftward node is parent.
                    parent_index = left - 1;
                    g_parent_index = g_left - 1;

                    // Current node is a right child.
                    child_addr = &(nodes[4 * g_parent_index + 0].y);
                    end_addr = &(nodes[4 * g_parent_index + 0].w);
                    end = -1 * (right + 1);

                    sm_addr = &(sm_nodes[parent_index - low].y);
                    sm_end = g_right;

                    // Right child AABB.
                    AABB_f4_addr = &(f4_nodes[4 * g_parent_index + 2].x);
                    AABB_f2_addr = &(f4_nodes[4 * g_parent_index + 3].z);

                }
                else {
                    // Rightward node is parent.
                    parent_index = right;
                    g_parent_index = g_right;

                    // Current node is a left child.
                    child_addr = &(nodes[4 * g_parent_index + 0].x);
                    end_addr = &(nodes[4 * g_parent_index + 0].z);
                    end = -1 * (left + 1);

                    sm_addr = &(sm_nodes[parent_index - low].x);
                    sm_end = g_left;

                    // Left child AABB.
                    AABB_f4_addr = &(f4_nodes[4 * g_parent_index + 1].x);
                    AABB_f2_addr = &(f4_nodes[4 * g_parent_index + 3].x);
                }

                if (parent_index < low || parent_index >= high) {
                    // Parent node outside this block's boundaries.  Either a
                    // thread in an adjacent block will follow this path, or the
                    // current node will be a base node in the next iteration.
                    break;
                }

                store_s32(child_addr, g_cur_index);
                store_s32(end_addr, end);
                store_vec4f32(AABB_f4_addr, x_min, x_max, y_min, y_max);
                store_vec2f32(AABB_f2_addr, z_min, z_max);
                *sm_addr = sm_end;

                __threadfence_block();
                assert(parent_index - low >= 0);
                assert(parent_index - low < BUILD_THREADS_PER_BLOCK + max_per_node);
                unsigned int flag = atomicAdd(&flags[parent_index - low], 1);
                assert(flag < 2);
                first_arrival = (flag == 0);
            } // while (!first_arrival)
        } // for idx = [threadIdx.x, BUILD_THREADS_PER_BLOCK + max_per_node)

        bid += gridDim.x;
        tid = threadIdx.x + bid * BUILD_THREADS_PER_BLOCK;
    } // while (bid * BUILD_THREADS_PER_BLOCK < n_base_nodes)
    return;
}

__global__ void fill_output_queue(const int4* nodes,
                                  const size_t n_nodes,
                                  const int max_per_node,
                                  int* new_base_indices)
{
    int tid = threadIdx.x + blockIdx.x * BUILD_THREADS_PER_BLOCK;

    while (tid < n_nodes)
    {
        int4 node = nodes[4 * tid + 0];

        // The left/right values are negative IFF they were written in this
        // iteration.  A negative index represents a logical/compacted index.
        bool left_written = (node.z < 0);
        bool right_written = (node.w < 0);

        // Undo the -ve encoding.
        int left = -1 * node.z - 1;
        int right = -1 * node.w - 1;

        if (left_written != right_written) {
            // Only one child was written; it must be placed in the output queue
            // at a *unique* location.
            int index = left_written ? left : right;
            assert(new_base_indices[index] == -1);
            new_base_indices[index] = left_written ? node.x : node.y;
        }
        else if (left_written) {
            // Both were written, so the current node must be added to the
            // output queue IFF it is sufficiently large; that is, if its size
            // is such that it would have caused the thread to end its tree
            // climb.
            // Again, the location we write to must be unique.
            int size = right - left + 1;
            if (size > max_per_node) {
                assert(new_base_indices[left] == -1);
                new_base_indices[left] = tid;
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void fix_node_ranges(int4* nodes,
                                const size_t n_nodes,
                                const int4* leaves,
                                const int* old_base_indices)
{
    int tid = threadIdx.x + blockIdx.x * BUILD_THREADS_PER_BLOCK;

    while (tid < n_nodes)
    {
        int4 node = nodes[4 * tid + 0];

        // We only need to fix nodes whose ranges were written, *in full*, this
        // iteration.
        if (node.z < 0 && node.w < 0)
        {
            int left = node.z < 0 ? (-1 * node.z - 1) : node.z;
            int right = node.w < 0 ? (-1 * node.w - 1) : node.w;

            // All base nodes have correct range indices.  If we know our
            // left/right-most base node, we can therefore find our
            // left/right-most leaf indices.
            // Note that for leaves, the left and right ranges are simply the
            // (corrected) index of the leaf.
            int index = old_base_indices[left];
            left = index < n_nodes ? nodes[4 * index + 0].z : index - n_nodes;

            index = old_base_indices[right];
            right = index < n_nodes ? nodes[4 * index + 0].w : index - n_nodes;

            // Only the left-most leaf can have index 0.
            assert(left >= 0);
            assert(left < n_nodes);
            assert(right > 0);
            // Only the right-most leaf can have index n_leaves - 1 == n_nodes.
            assert(right <= n_nodes);

            node.z = left;
            node.w = right;
            nodes[4 * tid + 0] = node;
        }

        tid += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tree building.
//-----------------------------------------------------------------------------

template<typename KeyType, typename DeltaType>
void compute_deltas(const thrust::device_vector<KeyType>& d_keys,
                    thrust::device_vector<DeltaType>& d_deltas)
{
    assert(d_keys.size() + 1 == d_deltas.size());

    int blocks = min(MAX_BLOCKS, (int)( (d_deltas.size() + 511) / 512 ));
    gpu::compute_deltas_kernel<<<blocks, 512>>>(
        thrust::raw_pointer_cast(d_keys.data()),
        d_keys.size(),
        thrust::raw_pointer_cast(d_deltas.data()));
}

template<typename KeyType, typename DeltaType>
void compute_leaf_deltas(const thrust::device_vector<int4>& d_leaves,
                         const thrust::device_vector<KeyType>& d_keys,
                         thrust::device_vector<DeltaType>& d_deltas)
{
    assert(d_leaves.size() + 1 == d_deltas.size());

    int blocks = min(MAX_BLOCKS, (int)( (d_leaves.size() + 511) / 512 ));
    gpu::compute_leaf_deltas_kernel<<<blocks, 512>>>(
        thrust::raw_pointer_cast(d_leaves.data()),
        d_leaves.size(),
        thrust::raw_pointer_cast(d_keys.data()),
        d_keys.size(),
        thrust::raw_pointer_cast(d_deltas.data()));
}

template <typename DeltaType>
void build_leaves(thrust::device_vector<int2>& d_tmp_nodes,
                  thrust::device_vector<int4>& d_tmp_leaves,
                  const int max_per_leaf,
                  const thrust::device_vector<DeltaType>& d_deltas)
{
    const size_t n_leaves = d_tmp_leaves.size();
    const size_t n_nodes = n_leaves - 1;


    int blocks = min(MAX_BLOCKS, (int) ((n_leaves + BUILD_THREADS_PER_BLOCK-1)
                                        / BUILD_THREADS_PER_BLOCK));
    int smem_size = sizeof(int) * (BUILD_THREADS_PER_BLOCK + max_per_leaf);

    gpu::build_leaves_kernel<<<blocks, BUILD_THREADS_PER_BLOCK, smem_size>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_deltas.data()),
        max_per_leaf);

    blocks = min(MAX_BLOCKS, (int) ((n_nodes + BUILD_THREADS_PER_BLOCK-1)
                                     / BUILD_THREADS_PER_BLOCK));

    gpu::write_leaves_kernel<<<blocks, BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_tmp_leaves.data()),
        max_per_leaf);
}

void remove_empty_leaves(Tree& d_tree)
{
    // A transform_reduce (with unary op 'is_valid_node()') followed by a
    // copy_if (with predicate 'is_valid_node()') actually seems slightly faster
    // than the below.  However, this method does not require a temporary leaves
    // array, which would be the largest temporary memory allocation in the
    // build process.

    // error: no operator "==" matches these operands
    //         operand types are: const int4 == const int4
    // thrust::remove(.., .., make_int4(0, 0, 0, 0))

    typedef thrust::device_vector<int4>::iterator Int4Iter;
    Int4Iter end = thrust::remove_if(d_tree.leaves.begin(), d_tree.leaves.end(),
                                     is_empty_node());

    const size_t n_new_leaves = end - d_tree.leaves.begin();
    const size_t n_new_nodes = n_new_leaves - 1;
    d_tree.nodes.resize(4 * n_new_nodes);
    d_tree.leaves.resize(n_new_leaves);
}

template <typename DeltaType, typename Float4>
void build_nodes(Tree& d_tree,
                 const thrust::device_vector<DeltaType>& d_deltas,
                 const thrust::device_vector<Float4>& d_spheres)
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

        int blocks = min(MAX_BLOCKS, (int) ((n_in + BUILD_THREADS_PER_BLOCK-1)
                                             / BUILD_THREADS_PER_BLOCK));
        // SMEM has to cover for BUILD_THREADS_PER_BLOCK + max_per_leaf flags
        // AND int2 nodes.
        int smem_size = (sizeof(int) + sizeof(int2))
                        * (BUILD_THREADS_PER_BLOCK + d_tree.max_per_leaf);
        gpu::build_nodes_slice_kernel<<<blocks, BUILD_THREADS_PER_BLOCK, smem_size>>>(
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            reinterpret_cast<float4*>(
                thrust::raw_pointer_cast(d_tree.nodes.data())),
            n_nodes,
            thrust::raw_pointer_cast(d_tree.leaves.data()),
            n_leaves,
            thrust::raw_pointer_cast(d_spheres.data()),
            d_in_ptr,
            n_in,
            d_tree.root_index_ptr,
            thrust::raw_pointer_cast(d_deltas.data()),
            d_tree.max_per_leaf, // This can actually be anything.
            d_out_ptr);

        blocks = min(MAX_BLOCKS, (int) ((n_nodes + BUILD_THREADS_PER_BLOCK-1)
                                             / BUILD_THREADS_PER_BLOCK));
        gpu::fill_output_queue<<<blocks, BUILD_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            n_nodes,
            d_tree.max_per_leaf,
            d_out_ptr);

        gpu::fix_node_ranges<<<blocks, BUILD_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_tree.nodes.data()),
            n_nodes,
            thrust::raw_pointer_cast(d_tree.leaves.data()),
            d_in_ptr);

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

template <typename KeyType, typename DeltaType, typename Float4>
void build_tree(Tree& d_tree,
                const thrust::device_vector<KeyType>& d_keys,
                const thrust::device_vector<DeltaType>& d_deltas,
                const thrust::device_vector<Float4>& d_spheres,
                const bool wipe = false)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.

    // In case this ever changes.
    assert(sizeof(int4) == sizeof(float4));

    if (wipe) {
        int4 empty = make_int4(0, 0, 0, 0);
        thrust::fill(d_tree.nodes.begin(), d_tree.nodes.end(), empty);
        thrust::fill(d_tree.leaves.begin(), d_tree.leaves.end(), empty);
    }

    const size_t n_leaves = d_tree.leaves.size();
    const size_t n_nodes = n_leaves - 1;

    thrust::device_vector<int2> d_tmp_nodes(n_nodes);

    build_leaves(d_tmp_nodes, d_tree.leaves, d_tree.max_per_leaf, d_deltas);
    remove_empty_leaves(d_tree);

    const size_t n_new_leaves = d_tree.leaves.size();

    thrust::device_vector<DeltaType> d_new_deltas(n_new_leaves + 1);
    compute_leaf_deltas(d_tree.leaves, d_keys, d_new_deltas);

    build_nodes(d_tree, d_new_deltas, d_spheres);
}

} // namespace grace
