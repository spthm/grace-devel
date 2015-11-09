#pragma once

#include "error.h"
#include "types.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "vector_types.h"

namespace grace {

class Tree
{
public:
    // A 32-bit int allows for indices up to 2,147,483,647 > 1024^3.
    // Since we likely can't fit more than 1024^3 particles on the GPU, int ---
    // which is 32-bit on relevant platforms --- should be sufficient for the
    // indices. In ALBVH there is no reason not to use uint, though.
    //
    // nodes[4*node_ID + 0].x: left child index
    //                     .y: right child index
    //                     .z: index of first leaf in this node
    //                     .w: index of the last leaf in this node
    // nodes[4*node_ID + 1].x = left_bx
    //                     .y = left_tx
    //                     .z = left_by
    //                     .w = left_ty
    // nodes[4*node_ID + 2].x = right_bx
    //                     .y = right_tx
    //                     .z = right_by
    //                     .w = right_ty
    // nodes[4*node_ID + 3].x = left_bz
    //                     .y = left_tz
    //                     .z = right_bz
    //                     .w = right_tz
    thrust::device_vector<int4> nodes;
    // leaves[leaf_ID].x = index of first sphere in this leaf
    //                .y = number of spheres in this leaf
    //                .z = padding
    //                .w = padding
    thrust::device_vector<int4> leaves;
    // A pointer to the *value of the index* of the root element of the tree.
    int* root_index_ptr;
    int max_per_leaf;

    Tree(size_t N_leaves, int max_per_leaf = 1) :
        nodes(4*(N_leaves-1)), leaves(N_leaves), max_per_leaf(max_per_leaf)
    {
       GRACE_CUDA_CHECK(cudaMalloc(&root_index_ptr, sizeof(int)));
    }

    ~Tree()
    {
        GRACE_CUDA_CHECK(cudaFree(root_index_ptr));
    }
};

class H_Tree
{
public:
    thrust::host_vector<int4> nodes;
    thrust::host_vector<int4> leaves;
    int root_index;
    int max_per_leaf;

    H_Tree(size_t N_leaves, int _max_per_leaf = 1) :
        nodes(4*(N_leaves-1)), leaves(N_leaves), root_index(0),
        max_per_leaf(max_per_leaf) {}
};


//-----------------------------------------------------------------------------
// Helper functions for tree build kernels.
//-----------------------------------------------------------------------------

struct is_empty_node : public thrust::unary_function<int4, bool>
{
    GRACE_HOST_DEVICE
    bool operator()(const int4 node) const
    {
        // Note: a node's right child can never be node 0, and a leaf can never
        // cover zero elements.
        return (node.y == 0);
    }
};

} //namespace grace
