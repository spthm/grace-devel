#pragma once

#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

#include <math_constants.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

//-----------------------------------------------------------------------------
// Helper functions for tree build kernels.
//-----------------------------------------------------------------------------

struct is_valid_node : public thrust::unary_function<int4, int>
{
    __host__ __device__
    int operator()(const int4 node) const
    {
        // Note: a node's right child can never be node 0, and a leaf can never
        // cover zero elements.
        return (node.y > 0);
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
__global__ void build_leaves_kernel(volatile int4* nodes,
                                    const size_t n_nodes,
                                    int4* big_leaves,
                                    const DeltaType* deltas,
                                    const int max_per_leaf,
                                    unsigned int* flags)
{
    int tid, cur_index, parent_index;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x * BUILD_THREADS_PER_BLOCK;
    const size_t n_leaves = n_nodes + 1;
    // Offset deltas so the range [-1, n_keys) is valid for indexing it.
    deltas++;

    while  (tid < n_leaves)
    {
        cur_index = tid;

        // Compute the current leaf's parent index and write associated data to
        // the parent. The leaf is not actually written.
        int left = cur_index;
        int right = cur_index;
        if (deltas[left - 1] < deltas[right])
        {
            // Leftward node is parent.
            parent_index = left - 1;
            // Current leaf is a right child.
            nodes[parent_index].w = right;
        }
        else {
            // Rightward node is parent.
            parent_index = right;
            // Current leaf is a left child.
            nodes[parent_index].z = left;
        }

        // Travel up the tree.  The second thread to reach a node writes
        // its AABB to its parent.  The first exits the loop.
        __threadfence();
        first_arrival = (atomicAdd(&flags[parent_index], 1) == 0);

        while (!first_arrival)
        {
            cur_index = parent_index;

            left = nodes[cur_index].z;
            right = nodes[cur_index].w;
            // Only the left-most leaf can have an index of 0, and only the
            // right-most leaf can have an index of n_leaves - 1.
            assert(left >= 0 && left < n_leaves - 1);
            assert(right > 0 && right < n_leaves);

            int size = right - left + 1;
            if (size > max_per_leaf) {
                // At least one child of the current node must be a leaf.
                // Stop traveling up the tree and continue with outer loop.
                break;
            }

            // Compute the current node's parent index and write associated
            // data.
            if (deltas[left - 1] < deltas[right])
            {
                // Leftward node is parent.
                parent_index = left - 1;
                // Current node is a right child.
                nodes[parent_index].w = right;
            }
            else {
                // Rightward node is parent.
                parent_index = right;
                // Current node is a left child.
                nodes[parent_index].z = left;
            }

            __threadfence();
            first_arrival = (atomicAdd(&flags[parent_index], 1) == 0);
        }
        tid += BUILD_THREADS_PER_BLOCK * gridDim.x;
    }
    return;
}

__global__ void fix_leaves_kernel(const int4* nodes,
                                  const size_t n_nodes,
                                  int4* big_leaves,
                                  const int max_per_leaf)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n_nodes)
    {
        // This node had at least one of its children written.
        int4 node = nodes[tid];
        int left = node.z;
        int right = node.w;

        // If left or right are 0, size may be incorrect.
        int size = right - left + 1;

        // If left is 0, left_size may be *incorrect*:
        // we cannot differentiate between an unwritten node.z and one
        // written as 0.
        int left_size = tid - left + 1;
        // This is guaranteed to be *sufficiently correct*:
        // right == 0 means node.w was unwritten, and the right child is
        // therefore not a leaf, so its size is set accordingly.
        int right_size = (right > 0 ? right - tid : max_per_leaf + 1);

        // These are both guarranteed to be *correct*:
        // If left_size was computed incorrectly, then the true value of
        // node.z is not zero, and thus node.z was unwritten. This requires
        // that the left child is *not* a leaf, and hence the node index
        // (tid) must be  >= max_per_leaf, resulting in left_leaf = false.
        // right_leaf follows from the correctness of right_size.
        bool left_leaf = (left_size <= max_per_leaf);
        bool right_leaf = (right_size <= max_per_leaf);

        // If only one child is to be written, we are certain it should be,
        // as the current node's size must be > max_per_leaf.
        // Otherwise: we write only if the current node cannot be a leaf.
        bool size_check = left_leaf ^ right_leaf ? true :
                                                   (size > max_per_leaf);

        // NOTE: size is guaranteed accurate only if both left_leaf and
        // right_leaf are true, but if they are both false no write occurs
        // anyway because of the && below.
        int4 leaf;
        if (left_leaf && size_check) {
            leaf.x = left;
            leaf.y = left_size;
            big_leaves[left] = leaf;
        }
        if (right_leaf && size_check) {
            leaf.x = tid + 1;
            leaf.y = right_size;
            big_leaves[right] = leaf;
        }

        tid += blockDim.x * gridDim.x;
    }
}

template <typename DeltaType, typename Float4>
__global__ void build_nodes_kernel(volatile int4* nodes,
                                   volatile float4* f4_nodes,
                                   int4* leaves,
                                   const size_t n_leaves,
                                   unsigned int* heights,
                                   int* root_index,
                                   const DeltaType* deltas,
                                   const Float4* spheres,
                                   unsigned int* flags)
{
    int tid, cur_index, parent_index;
    Float4 sphere;
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
    bool first_arrival;

    // Ensure the int4/float4 recasting of node data is valid.
    assert(sizeof(int4) == sizeof(float4));

    tid = threadIdx.x + blockIdx.x * BUILD_THREADS_PER_BLOCK;
    const size_t n_nodes = n_leaves - 1;
    // Offset deltas so the range [-1, n_keys) is valid for indexing it.
    deltas++;

    x_min = y_min = z_min = CUDART_INF_F;
    x_max = y_max = z_max = -1.f;
    while  (tid < n_leaves)
    {
        cur_index = tid;
        int4 leaf = leaves[cur_index];

        // Compute the current leaf's AABB.
        for (int i = 0; i < leaf.y; i++) {
            sphere = spheres[leaf.x + i];

            x_max = max(x_max, sphere.x + sphere.w);
            y_max = max(y_max, sphere.y + sphere.w);
            z_max = max(z_max, sphere.z + sphere.w);

            x_min = min(x_min, sphere.x - sphere.w);
            y_min = min(y_min, sphere.y - sphere.w);
            z_min = min(z_min, sphere.z - sphere.w);
        }

        // Compute the current leaf's parent index, write the updated leaf and
        // write this leaf's share of its parent's data.
        int left = cur_index;
        int right = cur_index;
        if (deltas[left - 1] < deltas[right])
        {
            // Leftward node is parent.
            parent_index = left - 1;
            leaf.z = parent_index;

            // Current leaf is a right child.
            nodes[4 * parent_index + 0].y = cur_index + n_nodes;
            nodes[4 * parent_index + 0].w = right;

            // Write current node's AABB (to its parent).
            f4_nodes[4 * parent_index + 1].z = x_min;
            f4_nodes[4 * parent_index + 1].w = x_max;
            f4_nodes[4 * parent_index + 2].z = y_min;
            f4_nodes[4 * parent_index + 2].w = y_max;
            f4_nodes[4 * parent_index + 3].z = z_min;
            f4_nodes[4 * parent_index + 3].w = z_max;

        }
        else {
            // Rightward node is parent.
            parent_index = right;
            leaf.z = parent_index;

            // Current leaf is a left child.
            nodes[4 * parent_index + 0].x = cur_index + n_nodes;
            nodes[4 * parent_index + 0].z = left;

            // Write current node's AABB (to its parent).
            f4_nodes[4 * parent_index + 1].x = x_min;
            f4_nodes[4 * parent_index + 1].y = x_max;
            f4_nodes[4 * parent_index + 2].x = y_min;
            f4_nodes[4 * parent_index + 2].y = y_max;
            f4_nodes[4 * parent_index + 3].x = z_min;
            f4_nodes[4 * parent_index + 3].y = z_max;
        }

        // TODO: We don't actually need to store this information.
        leaves[cur_index] = leaf;

        // Travel up the tree.  The second thread to reach a node writes
        // its AABB to its parent.  The first exits the loop.
        int height = 0;
        __threadfence();

        unsigned int flag = atomicAdd(&flags[parent_index], 1);
        assert(flag < 2);
        first_arrival = (flag == 0);

        while (!first_arrival)
        {
            cur_index = parent_index;

            heights[cur_index] = height;
            height++;

            left = nodes[4 * cur_index + 0].z;
            right = nodes[4 * cur_index + 0].w;
            // Only the left-most leaf can have an index of 0, and only the
            // right-most leaf can have an index of n_leaves - 1.
            assert(left >= 0 && left < n_leaves - 1);
            assert(right > 0 && right < n_leaves);

            int size = right - left;
            if (size == n_nodes) {
                // A second thread has reached the root node, so all nodes
                // have now been processed.
                *root_index = cur_index;
                return;
            }

            // Compute the current node's AABB as the union of its children's
            // AABBs.
            x_min = min(f4_nodes[4 * cur_index + 1].x,
                        f4_nodes[4 * cur_index + 1].z);
            x_max = max(f4_nodes[4 * cur_index + 1].y,
                        f4_nodes[4 * cur_index + 1].w);
            y_min = min(f4_nodes[4 * cur_index + 2].x,
                        f4_nodes[4 * cur_index + 2].z);
            y_max = max(f4_nodes[4 * cur_index + 2].y,
                        f4_nodes[4 * cur_index + 2].w);
            z_min = min(f4_nodes[4 * cur_index + 3].x,
                        f4_nodes[4 * cur_index + 3].z);
            z_max = max(f4_nodes[4 * cur_index + 3].y,
                        f4_nodes[4 * cur_index + 3].w);

            // Note, they should never be equal.
            assert(x_min < x_max);
            assert(y_min < y_max);
            assert(z_min < z_max);

            // Compute the current node's parent index and write associated
            // data.
            if (deltas[left - 1] < deltas[right])
            {
                // Leftward node is parent.
                parent_index = left - 1;

                // Current node is a right child.
                nodes[4 * parent_index + 0].y = cur_index;
                nodes[4 * parent_index + 0].w = right;

                // Write current node's AABB (to its parent).
                f4_nodes[4 * parent_index + 1].z = x_min;
                f4_nodes[4 * parent_index + 1].w = x_max;
                f4_nodes[4 * parent_index + 2].z = y_min;
                f4_nodes[4 * parent_index + 2].w = y_max;
                f4_nodes[4 * parent_index + 3].z = z_min;
                f4_nodes[4 * parent_index + 3].w = z_max;

            }
            else {
                // Rightward node is parent.
                parent_index = right;

                // Current node is a left child.
                nodes[4 * parent_index + 0].x = cur_index;
                nodes[4 * parent_index + 0].z = left;

                // Write current node's AABB (to its parent).
                f4_nodes[4 * parent_index + 1].x = x_min;
                f4_nodes[4 * parent_index + 1].y = x_max;
                f4_nodes[4 * parent_index + 2].x = y_min;
                f4_nodes[4 * parent_index + 2].y = y_max;
                f4_nodes[4 * parent_index + 3].x = z_min;
                f4_nodes[4 * parent_index + 3].y = z_max;
            }

            __threadfence();
            unsigned int flag = atomicAdd(&flags[parent_index], 1);
            assert(flag < 2);
            first_arrival = (flag == 0);
        }
        tid += BUILD_THREADS_PER_BLOCK * gridDim.x;
    }
    return;
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
void build_leaves(thrust::device_vector<int4>& d_tmp_nodes,
                  thrust::device_vector<int4>& d_tmp_leaves,
                  const int max_per_leaf,
                  const thrust::device_vector<DeltaType>& d_deltas,
                  thrust::device_vector<unsigned int>& d_flags)
{
    const size_t n_leaves = d_tmp_leaves.size();
    const size_t n_nodes = n_leaves - 1;

    int blocks = min(MAX_BLOCKS, (int) ((n_leaves + BUILD_THREADS_PER_BLOCK-1)
                                        / BUILD_THREADS_PER_BLOCK));

    gpu::build_leaves_kernel<<<blocks, BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_tmp_leaves.data()),
        thrust::raw_pointer_cast(d_deltas.data()),
        max_per_leaf,
        thrust::raw_pointer_cast(d_flags.data()));

    blocks = min(MAX_BLOCKS, (int) ((n_nodes + BUILD_THREADS_PER_BLOCK-1)
                                     / BUILD_THREADS_PER_BLOCK));

    gpu::fix_leaves_kernel<<<blocks, BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tmp_nodes.data()),
        n_nodes,
        thrust::raw_pointer_cast(d_tmp_leaves.data()),
        max_per_leaf);
}

template <typename DeltaType, typename Float4>
void build_nodes(Tree& d_tree,
                 const thrust::device_vector<DeltaType>& d_deltas,
                 const thrust::device_vector<Float4>& d_spheres,
                 thrust::device_vector<unsigned int>& d_flags)
{
    const size_t n_leaves = d_tree.leaves.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_leaves + BUILD_THREADS_PER_BLOCK-1)
                                         / BUILD_THREADS_PER_BLOCK));

    gpu::build_nodes_kernel<<<blocks,BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tree.nodes.data()),
         reinterpret_cast<float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        n_leaves,
        thrust::raw_pointer_cast(d_tree.heights.data()),
        d_tree.root_index_ptr,
        thrust::raw_pointer_cast(d_deltas.data()),
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_flags.data()));
}

void copy_big_leaves(Tree& d_tree,
                     thrust::device_vector<int4>& d_tmp_leaves)
{
    // TODO: Since the copy_if presumably does a scan sum internally before
    // performing the copy, this is wasteful.  Calling a scan sum and then
    // doing the copy 'manually' would probably be better.
    // Alternatively, try thrust::remove_copy with value = int4().
    // Not using a temporary leaves array above, and using thrust::remove
    // here is also an option (d_tree.nodes could be used as the first argument
    // to build_tree_leaves_kernel as all its data will be overwritten later
    // anyway.)
    const int n_new_leaves = thrust::transform_reduce(d_tmp_leaves.begin(),
                                                      d_tmp_leaves.end(),
                                                      is_valid_node(),
                                                      0,
                                                      thrust::plus<int>());
    const int n_new_nodes = n_new_leaves - 1;
    d_tree.nodes.resize(4 * n_new_nodes);
    d_tree.leaves.resize(n_new_leaves);

    thrust::copy_if(d_tmp_leaves.begin(), d_tmp_leaves.end(),
                    d_tree.leaves.begin(),
                    is_valid_node());
}

template <typename KeyType, typename DeltaType, typename Float4>
void build_tree(Tree& d_tree,
                const thrust::device_vector<KeyType>& d_keys,
                const thrust::device_vector<DeltaType>& d_deltas,
                const thrust::device_vector<Float4>& d_spheres)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.

    // In case this ever changes.
    assert(sizeof(int4) == sizeof(float4));

    const size_t n_leaves = d_tree.leaves.size();
    const size_t n_nodes = n_leaves - 1;

    thrust::device_vector<unsigned int> d_flags(n_nodes);
    thrust::device_vector<int4> d_tmp_leaves(n_leaves);

    // Use d_tree.leaves as temporary *nodes* here (working memory for the
    // partial tree climb).
    // d_tmp_leaves stores what will become the final leaf nodes.
    build_leaves(d_tree.leaves, d_tmp_leaves, d_tree.max_per_leaf, d_deltas,
                 d_flags);

    copy_big_leaves(d_tree, d_tmp_leaves);

    const size_t n_new_leaves = d_tree.leaves.size();
    const size_t n_new_nodes = n_new_leaves - 1;

    thrust::device_vector<DeltaType> d_new_deltas(n_new_leaves + 1);
    compute_leaf_deltas(d_tree.leaves, d_keys, d_new_deltas);

    d_flags.resize(n_new_nodes);
    thrust::fill(d_flags.begin(), d_flags.end(), 0);
    build_nodes(d_tree, d_new_deltas, d_spheres, d_flags);
}

} // namespace grace
