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

// From thrust/examples.strided_range.cu, author Nathan Bell.
template <typename Iterator>
class strided_iterator
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_iterator iterator
    typedef PermutationIterator iterator;

    // construct strided_iterator for the range [first,last)
    strided_iterator(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

struct is_valid_node
{
    __host__ __device__
    int operator()(const int4 node) const
    {
        // Implementation note: during compaction, node.y < 0 is a valid index.
        return (node.y != 0);
    }

    template <typename NodeType, typename AABBType>
    __host__ __device__
    int operator()(
        const thrust::tuple<NodeType, AABBType, AABBType, AABBType> node_tuple)
    const
    {
        NodeType node = thrust::get<0>(node_tuple);
        return (node.y != 0);
    }
};

struct flag_invalid : public thrust::unary_function<int4, int>
{
    __host__ __device__
    int operator()(const int4 node) const
    {
        return !(is_valid_node()(node));
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

template <typename InType, typename DeltaType>
__global__ void compute_deltas_kernel(const InType* input,
                                      const size_t n_input,
                                      DeltaType* deltas)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid <= n_input)
    {
        // The range [-1, n_keys) is valid for querying node_delta.
        deltas[tid] = node_delta(tid - 1, input, n_input);
        tid += blockDim.x * gridDim.x;
    }
}

template <typename DeltaType, typename Float4>
__global__ void build_tree_kernel(volatile int4* nodes,
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

    while  (tid < n_leaves)
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
        assert(left >= 0);
        assert(right < n_leaves);
        if (deltas[left - 1] < deltas[right])
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
        int height = 0;
        __threadfence();
        first_arrival = (atomicAdd(&flags[parent_index], 1) == 0);

        while (!first_arrival)
        {
            cur_index = parent_index;

            heights[cur_index] = height;
            height++;

            left = nodes[4 * cur_index + 0].z;
            right = nodes[4 * cur_index + 0].w;
            assert(left >= 0);
            assert(right < n_leaves);
            if (right - left == n_nodes) {
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
            first_arrival = (atomicAdd(&flags[parent_index], 1) == 0);
        }
        tid += BUILD_THREADS_PER_BLOCK * gridDim.x;
    }
    return;
}

__global__ void wipe_invalids_kernel(int4* nodes,
                                     const size_t n_nodes,
                                     int4* new_leaves,
                                     const int max_per_leaf)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n_nodes)
    {
        int4 node = nodes[4 * tid];
        int node_size = node.w - node.z + 1;

        if (node_size <= max_per_leaf) {
            // Node is too small, wipe.
            node = int4();
        }
        else {
            // Check if either of the child nodes should become leaves.
            int left_size = tid - node.z + 1;
            int right_size = node.w - tid;

            if (left_size <= max_per_leaf) {
                int4 leaf;
                leaf.x = node.z; // First sphere index.
                leaf.y = left_size; // Sphere count.
                leaf.z = tid; // Parent.

                int index = (node.x >= n_nodes ? node.x - n_nodes : node.x);
                new_leaves[index] = leaf;

                // If the child is a node becoming a new leaf, mark its index
                // as such.  +1 accounts for node.x = 0;
                node.x = (node.x < n_nodes ? -1 * (node.x + 1) : node.x);
            }
            if (right_size <= max_per_leaf) {
                int4 leaf;
                leaf.x = tid + 1;
                leaf.y = right_size;
                leaf.z = tid;

                int index = (node.y >= n_nodes ? node.y - n_nodes : node.y);
                new_leaves[index] = leaf;

                node.y = (node.y < n_nodes ? -1 * node.y : node.y); // node.y cannot be 0, no +1.

            }
        }
        nodes[4 * tid] = node;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void fix_indices_kernel(int4* nodes, int4* leaves,
                                   size_t n_new_nodes, int* node_shifts,
                                   size_t n_old_nodes,
                                   int* root_index)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        int4 leaf = leaves[n_new_nodes];
        int shift = node_shifts[leaf.z];
        leaf.z -= shift;
        leaves[n_new_nodes] = leaf;

        *root_index -= node_shifts[*root_index];
    }

    while (tid < n_new_nodes)
    {
        int4 leaf = leaves[tid];
        int4 node = nodes[4 * tid];

        bool left_to_leaf = false;
        bool right_to_leaf = false;

        // Old nodes becoming new leaves are identified with negative indices.
        // The -1 allows for old node 0 to become a leaf.
        if (node.x < 0) {
            node.x = -1 * node.x - 1;
            left_to_leaf = true;
        }
        if (node.y < 0) {
            node.y *= -1; // A right child cannot be at index 0, no -1.
            right_to_leaf = true;
        }

        // Old leaves becoming new leaves need to have their leaf-identifying
        // offset removed so we can index node_shifts with them.
        // Note than these must occur *AFTER* the < 0 comparisons since
        // n_old_nodes is unsigned and an int-unsigned comparison would not be
        // safe for negative ints.
        if (node.x >= n_old_nodes) {
            node.x -= n_old_nodes;
            left_to_leaf = true;
        }
        if (node.y >= n_old_nodes) {
            node.y -= n_old_nodes;
            right_to_leaf = true;
        }

        // Leaf's parent must be a node: the shift is simple.
        int parent_shift = node_shifts[leaf.z];

        // An old node becoming a new leaf requires the shift NOT INCLUDING
        // itself, which is the offset of the previous node.
        // An old node becoming a new node has a shift which is equal to the
        // previous shift (it was an inclusive sum).
        // An old leaf becoming a new leaf takes the shift of its parent which,
        // as directly above, is equal to the previous shift.
        // node.x == 0 is a special case, necessarily having a shift of zero.
        int shift_left = node.x > 0 ? node_shifts[node.x - 1] : 0;
        int shift_right = node_shifts[node.y - 1];

        leaf.z -= parent_shift;

        // Child nodes which are leaves requiring an identifying offset.
        if (left_to_leaf)
            node.x += n_new_nodes;
        if (right_to_leaf)
            node.y += n_new_nodes;

        node.x -= shift_left;
        node.y -= shift_right;

        leaves[tid] = leaf;
        nodes[4 * tid] = node;

        tid += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tree building.
//-----------------------------------------------------------------------------

template<typename InType, typename DeltaType>
void compute_deltas(const thrust::device_vector<InType>& d_input,
                    thrust::device_vector<DeltaType>& d_deltas)
{
    const size_t n_input = d_input.size();
    assert(n_input + 1 == d_deltas.size());

    int blocks = min(MAX_BLOCKS, (int)( (d_deltas.size() + 511) / 512 ));
    gpu::compute_deltas_kernel<<<blocks, 512>>>(
        thrust::raw_pointer_cast(d_input.data()),
        n_input,
        thrust::raw_pointer_cast(d_deltas.data()));
}

template <typename DeltaType, typename Float4>
void build_tree(Tree& d_tree,
                const thrust::device_vector<DeltaType>& d_deltas,
                const thrust::device_vector<Float4>& d_spheres)
{
    // TODO: Error if n_keys <= 1 OR n_keys > MAX_INT.

    // In case this ever changes.
    assert(sizeof(int4) == sizeof(float4));

    const size_t n_leaves = d_tree.leaves.size();
    thrust::device_vector<unsigned int> d_flags(n_leaves-1);

    int blocks = min(MAX_BLOCKS, (int) ((n_leaves + BUILD_THREADS_PER_BLOCK-1)
                                        / BUILD_THREADS_PER_BLOCK));

    gpu::build_tree_kernel<<<blocks,BUILD_THREADS_PER_BLOCK>>>(
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

void compact_tree(Tree& d_tree)
{
    typedef thrust::device_vector<int4>::iterator Int4Iter;

    size_t n_old_leaves = d_tree.leaves.size();
    size_t n_old_nodes = n_old_leaves - 1;

    // Wipe node which span <= max_per_leaf values.
    // If a node spans > max_per_leaf values, write any child nodes which
    // are small enough be become leaves to big_leaves.
    // (big_leaves will have gaps.)
    thrust::device_vector<int4> big_leaves(n_old_leaves);
    thrust::device_vector<int4> old_nodes = d_tree.nodes;

    int blocks = min(MAX_BLOCKS,
                    (int) ((n_old_nodes + BUILD_THREADS_PER_BLOCK-1)
                          / BUILD_THREADS_PER_BLOCK));

    gpu::wipe_invalids_kernel<<<blocks, BUILD_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(old_nodes.data()),
        n_old_nodes,
        thrust::raw_pointer_cast(big_leaves.data()),
        d_tree.max_per_leaf);

    // Get the inclusive sum of the invalid nodes.
    // node_shifts[i] is how many nodes have been removed up to and *including*
    // node i.
    thrust::device_vector<int> node_shifts(n_old_nodes);
    strided_iterator<Int4Iter> old_hierarchy_iter(old_nodes.begin(),
                                                  old_nodes.end(),
                                                  4);

    thrust::transform_inclusive_scan(old_hierarchy_iter.begin(),
                                     old_hierarchy_iter.end(),
                                     node_shifts.begin(),
                                     flag_invalid(),
                                     thrust::plus<int>());

    // Compute the new tree sizes and resize the tree vectors.
    int n_new_nodes = n_old_nodes - node_shifts.back();;
    int n_new_leaves = n_new_nodes + 1;

    d_tree.nodes.resize(4 * n_new_nodes);
    d_tree.leaves.resize(n_new_leaves);

    // Copy over only the new leaves that have been written.
    // Except for their parent indices, the leaves are now compacted.
    thrust::copy_if(big_leaves.begin(), big_leaves.end(),
                    d_tree.leaves.begin(),
                    is_valid_node());

    // As above, but for the nodes.
    // A strided iterator is required to access every 4th element in the nodes
    // vector --- the elements that specify the node's children and size ---
    // and check its validity.
    // We copy the AABBs at the same time by using a zipped iterator.
    // Each element in the zipped iterator is a tuple, and the tuple contains
    // all the elements for a particular node (children and size, and the three
    // AABB float4s.)
    // AABBs are technically float4s, but the d_tree.nodes iterators are for
    // int4s; since the sizes are the same, we just pretend they are int4s.
    strided_iterator<Int4Iter> hierarchy_iter(d_tree.nodes.begin(),
                                              d_tree.nodes.end(), 4);
    strided_iterator<Int4Iter> AABBX_iter(d_tree.nodes.begin()+1,
                                          d_tree.nodes.end(), 4);
    strided_iterator<Int4Iter> AABBY_iter(d_tree.nodes.begin()+2,
                                          d_tree.nodes.end(), 4);
    strided_iterator<Int4Iter> AABBZ_iter(d_tree.nodes.begin()+3,
                                          d_tree.nodes.end(), 4);

    strided_iterator<Int4Iter> old_AABBX_iter(old_nodes.begin()+1,
                                              old_nodes.end(), 4);
    strided_iterator<Int4Iter> old_AABBY_iter(old_nodes.begin()+2,
                                              old_nodes.end(), 4);
    strided_iterator<Int4Iter> old_AABBZ_iter(old_nodes.begin()+3,
                                              old_nodes.end(), 4);

    typedef strided_iterator<Int4Iter>::iterator NodeIter;
    // Tuple representing a node.
    typedef thrust::tuple<NodeIter, NodeIter, NodeIter, NodeIter>
        NodeTupleIter;
    typedef thrust::zip_iterator<NodeTupleIter> NodesZipIter;

    NodesZipIter nodes_iter_begin(
        thrust::make_tuple(hierarchy_iter.begin(),
                           AABBX_iter.begin(),
                           AABBY_iter.begin(),
                           AABBZ_iter.begin()));

    NodesZipIter old_iter_begin(
        thrust::make_tuple(old_hierarchy_iter.begin(),
                           old_AABBX_iter.begin(),
                           old_AABBY_iter.begin(),
                           old_AABBZ_iter.begin()));

    NodesZipIter old_iter_end(
        thrust::make_tuple(old_hierarchy_iter.end(),
                           old_AABBX_iter.end(),
                           old_AABBY_iter.end(),
                           old_AABBZ_iter.end()));

    thrust::copy_if(old_iter_begin, old_iter_end,
                    nodes_iter_begin,
                    is_valid_node());

    blocks = min(MAX_BLOCKS,
                (int) ((n_new_nodes + SHIFTS_THREADS_PER_BLOCK-1)
                      / SHIFTS_THREADS_PER_BLOCK));

    // Fix all parent and child index references.
    gpu::fix_indices_kernel<<<blocks, SHIFTS_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_tree.nodes.data()),
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        n_new_nodes,
        thrust::raw_pointer_cast(node_shifts.data()),
        n_old_nodes,
        d_tree.root_index_ptr);
}

} // namespace grace
