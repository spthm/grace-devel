#pragma once

#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform_scan.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

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

//-----------------------------------------------------------------------------
// CUDA kernels for tree building.
//-----------------------------------------------------------------------------

template <typename KeyType, typename Float4>
__global__ void build_tree_kernel(volatile int4* nodes,
                                  volatile float4* f4_nodes,
                                  const size_t n_nodes,
                                  int4* leaves,
                                  unsigned int* levels,
                                  int* root_index,
                                  const KeyType* keys,
                                  const size_t n_keys,
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

    while  (tid < n_keys)
    {
        cur_index = tid;

        // Compute the current leaf's AABB.
        sphere = spheres[cur_index];

        x_max = sphere.x + sphere.w;
        y_max = sphere.y + sphere.w;
        z_max = sphere.z + sphere.w;

        x_min = sphere.x - sphere.w;
        y_min = sphere.y - sphere.w;
        z_min = sphere.z - sphere.w;

        // Compute the current leaf's parent index and write associated data.
        int left = cur_index;
        int right = cur_index;
        if (node_delta(left - 1, keys, n_keys)
                < node_delta(right, keys, n_keys))
        {
            // Leftward node is parent.
            parent_index = left - 1;
            leaves[cur_index].x = cur_index;
            leaves[cur_index].y = 1;
            leaves[cur_index].z = parent_index;

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
            leaves[cur_index].x = cur_index;
            leaves[cur_index].y = 1;
            leaves[cur_index].z = parent_index;

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

        // Travel up the tree.  The second thread to reach a node writes
        // its AABB to its parent.  The first exits the loop.
        int level = 32;
        __threadfence();
        first_arrival = (atomicAdd(&flags[parent_index], 1) == 0);

        while (!first_arrival)
        {
            cur_index = parent_index;

            levels[cur_index] = level;
            level--;

            left = nodes[4 * cur_index + 0].z;
            right = nodes[4 * cur_index + 0].w;
            if (right - left == n_keys - 1) {
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
            if (node_delta(left - 1, keys, n_keys)
                    < node_delta(right, keys, n_keys))
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
            first_arrival = (atomicAdd(&flags[parent_index], 1) == 0);
        }
        tid += BUILD_THREADS_PER_BLOCK * gridDim.x;
    }
    return;
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tree building.
//-----------------------------------------------------------------------------

template <typename KeyType, typename Float4>
void build_tree(Tree& d_tree,
                const thrust::device_vector<KeyType>& d_keys,
                const thrust::device_vector<Float4>& d_spheres)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.

    // In case this ever changes.
    assert(sizeof(int4) == sizeof(float4));

    size_t n_nodes = d_tree.leaves.size() - 1;
    size_t n_keys = d_keys.size();
    thrust::device_vector<unsigned int> d_flags(n_keys-1);

    int blocks = min(MAX_BLOCKS, (int) ((n_keys + BUILD_THREADS_PER_BLOCK-1)
                                        / BUILD_THREADS_PER_BLOCK));

    gpu::build_tree_kernel<<<blocks,BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tree.nodes.data()),
         reinterpret_cast<float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        thrust::raw_pointer_cast(d_tree.levels.data()),
        d_tree.root_index_ptr,
        thrust::raw_pointer_cast(d_keys.data()),
        n_keys,
        thrust::raw_pointer_cast(d_spheres.data()),
        thrust::raw_pointer_cast(d_flags.data()));
}

} // namespace grace
