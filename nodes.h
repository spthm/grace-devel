#pragma once

#include "device/loadstore.cuh"
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
// Helper functors for tree build kernels.
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

//-----------------------------------------------------------------------------
// Helper functions for node accesses.
//-----------------------------------------------------------------------------

GRACE_HOST_DEVICE bool is_leaf(int index, size_t n_nodes)
{
    return index >= n_nodes;
}

GRACE_HOST_DEVICE int4 get_inner(int index, const int4* nodes)
{
    return nodes[4 * index];
}
GRACE_DEVICE int4 get_inner(int index, volatile int4* nodes)
{
    return load_volatile_vec4s32(nodes + 4 * index);
}
GRACE_HOST_DEVICE int4 get_inner(int index, const float4* nodes)
{
    return get_inner(index, reinterpret_cast<const int4*>(nodes));
}
GRACE_DEVICE int4 get_inner(int index, volatile float4* nodes)
{
    float4 f4_node = load_volatile_vec4f32(nodes + 4 * index);
    return reinterpret_cast<int4&>(f4_node);
}

GRACE_HOST_DEVICE int4 get_leaf(int index, const int4* leaves)
{
    return leaves[index];
}

GRACE_HOST_DEVICE int4 get_node(int index, const int4* nodes, const int4* leaves, size_t n_nodes)
{
    if (is_leaf(index, n_nodes)) return get_leaf(index - n_nodes, leaves);
    else                         return get_inner(index, nodes);
}

GRACE_HOST_DEVICE float4 get_AABB1(int index, const float4* nodes)
{
    return nodes[4 * index + 1];
}
GRACE_DEVICE float4 get_AABB1(int index, volatile float4* nodes)
{
    return load_volatile_vec4f32(nodes + 4 * index + 1);
}
GRACE_HOST_DEVICE float4 get_AABB1(int index, const int4* nodes)
{
    return get_AABB1(index, reinterpret_cast<const float4*>(nodes));
}
GRACE_DEVICE float4 get_AABB1(int index, volatile int4* nodes)
{
    int4 i4_node = load_volatile_vec4s32(nodes + 4 * index + 1);
    return reinterpret_cast<float4&>(i4_node);
}

GRACE_HOST_DEVICE float4 get_AABB2(int index, const float4* nodes)
{
    return nodes[4 * index + 2];
}
GRACE_DEVICE float4 get_AABB2(int index, volatile float4* nodes)
{
    return load_volatile_vec4f32(nodes + 4 * index + 2);
}
GRACE_HOST_DEVICE float4 get_AABB2(int index, const int4* nodes)
{
    return get_AABB2(index, reinterpret_cast<const float4*>(nodes));
}
GRACE_DEVICE float4 get_AABB2(int index, volatile int4* nodes)
{
    int4 i4_node = load_volatile_vec4s32(nodes + 4 * index + 2);
    return reinterpret_cast<float4&>(i4_node);
}

GRACE_HOST_DEVICE float4 get_AABB3(int index, const float4* nodes)
{
    return nodes[4 * index + 3];
}
GRACE_DEVICE float4 get_AABB3(int index, volatile float4* nodes)
{
    return load_volatile_vec4f32(nodes + 4 * index + 3);
}
GRACE_HOST_DEVICE float4 get_AABB3(int index, const int4* nodes)
{
    return get_AABB3(index, reinterpret_cast<const float4*>(nodes));
}
GRACE_DEVICE float4 get_AABB3(int index, volatile int4* nodes)
{
    int4 i4_node = load_volatile_vec4s32(nodes + 4 * index + 3);
    return reinterpret_cast<float4&>(i4_node);
}

GRACE_HOST_DEVICE void get_left_AABB(int index, const float4* nodes,
                                     float3* bottom, float3* top)
{
    float4 AABBL  = get_AABB1(index, nodes);
    float4 AABBLR = get_AABB3(index, nodes);
    bottom->x = AABBL.x;
    top->x    = AABBL.y;
    bottom->y = AABBL.z;
    top->y    = AABBL.w;
    bottom->z = AABBLR.x;
    top->z    = AABBLR.y;
}
GRACE_HOST_DEVICE void get_left_AABB(int index, const int4* nodes,
                                     float3* bottom, float3* top)
{
    get_left_AABB(index, reinterpret_cast<const float4*>(nodes), bottom, top);
}

GRACE_HOST_DEVICE void get_right_AABB(int index, const float4* nodes,
                                      float3* bottom, float3* top)
{
    float4 AABBR  = get_AABB2(index, nodes);
    float4 AABBLR = get_AABB3(index, nodes);
    bottom->x = AABBR.x;
    top->x    = AABBR.y;
    bottom->y = AABBR.z;
    top->y    = AABBR.w;
    bottom->z = AABBLR.z;
    top->z    = AABBLR.w;
}
GRACE_HOST_DEVICE void get_right_AABB(int index, const int4* nodes,
                                      float3* bottom, float3* top)
{
    get_right_AABB(index, reinterpret_cast<const float4*>(nodes), bottom, top);
}

GRACE_HOST_DEVICE void set_inner(int4 node, int index, int4* nodes)
{
    nodes[4 * index] = node;
}
GRACE_DEVICE void set_inner(int4 node, int index, volatile int4* nodes)
{
    store_volatile_vec4s32(nodes + 4 * index, node);
}
GRACE_HOST_DEVICE void set_inner(int4 node, int index, float4* nodes)
{
    set_inner(node, index, reinterpret_cast<int4*>(nodes));
}
GRACE_DEVICE void set_inner(int4 node, int index, volatile float4* nodes)
{
    set_inner(node, index, reinterpret_cast<volatile int4*>(nodes));
}

GRACE_HOST_DEVICE void set_leaf(int4 leaf, int index, int4* leaves)
{
    leaves[index] = leaf;
}

GRACE_HOST_DEVICE void set_node(int4 node, int index, int4* nodes, int4* leaves,
                                size_t n_nodes)
{
    if (is_leaf(index, n_nodes)) set_leaf(node, index - n_nodes, leaves);
    else                         set_inner(node, index, nodes);
}

GRACE_HOST_DEVICE void set_left_node(int left_child, int left_leaf,
                                     int index, int4* nodes)
{
    nodes[4 * index].x = left_child;
    nodes[4 * index].z = left_leaf;
}
GRACE_DEVICE void set_left_node(int left_child, int left_leaf,
                                int index, volatile int4* nodes)
{
    nodes[4 * index].x = left_child;
    nodes[4 * index].z = left_leaf;
}
GRACE_HOST_DEVICE void set_left_node(int left_child, int left_leaf,
                                     int index, float4* nodes)
{
    set_left_node(left_child, left_leaf, index,
                  reinterpret_cast<int4*>(nodes));
}
GRACE_DEVICE void set_left_node(int left_child, int left_leaf,
                                int index, volatile float4* nodes)
{
    set_left_node(left_child, left_leaf, index,
                  reinterpret_cast<volatile int4*>(nodes));
}

GRACE_HOST_DEVICE void set_right_node(int right_child, int right_leaf,
                                      int index, int4* nodes)
{
    nodes[4 * index].y = right_child;
    nodes[4 * index].w = right_leaf;
}
GRACE_DEVICE void set_right_node(int right_child, int right_leaf,
                                 int index, volatile int4* nodes)
{
    nodes[4 * index].y = right_child;
    nodes[4 * index].w = right_leaf;
}
GRACE_HOST_DEVICE void set_right_node(int right_child, int right_leaf,
                                      int index, float4* nodes)
{
    set_right_node(right_child, right_leaf, index,
                   reinterpret_cast<int4*>(nodes));
}
GRACE_DEVICE void set_right_node(int right_child, int right_leaf,
                                 int index, volatile float4* nodes)
{
    set_right_node(right_child, right_leaf, index,
                   reinterpret_cast<volatile int4*>(nodes));
}

GRACE_HOST_DEVICE void set_left_AABB(float3 bottom, float3 top,
                                     int index, float4* nodes)
{
    nodes[4 * index + 1] = make_float4(bottom.x, top.x, bottom.y, top.y);
    nodes[4 * index + 3].x = bottom.z;
    nodes[4 * index + 3].y = top.z;
}
GRACE_DEVICE void set_left_AABB(float3 bottom, float3 top,
                                int index, volatile float4* nodes)
{
    float4 AABB1 = make_float4(bottom.x, top.x, bottom.y, top.y);
    store_volatile_vec4f32(nodes + 4 * index + 1, AABB1);
    nodes[4 * index + 3].x = bottom.z;
    nodes[4 * index + 3].y = top.z;
}
GRACE_HOST_DEVICE void set_left_AABB(float3 bottom, float3 top,
                                     int index, int4* nodes)
{
    set_left_AABB(bottom, top, index, reinterpret_cast<float4*>(nodes));
}
GRACE_DEVICE void set_left_AABB(float3 bottom, float3 top,
                                int index, volatile int4* nodes)
{
    set_left_AABB(bottom, top, index, reinterpret_cast<volatile float4*>(nodes));
}

GRACE_HOST_DEVICE void set_right_AABB(float3 bottom, float3 top,
                                      int index, float4* nodes)
{
    nodes[4 * index + 2] = make_float4(bottom.x, top.x, bottom.y, top.y);
    nodes[4 * index + 3].z = bottom.z;
    nodes[4 * index + 3].w = top.z;
}
GRACE_DEVICE void set_right_AABB(float3 bottom, float3 top,
                                 int index, volatile float4* nodes)
{
    float4 AABB2 = make_float4(bottom.x, top.x, bottom.y, top.y);
    store_volatile_vec4f32(nodes + 4 * index + 2, AABB2);
    nodes[4 * index + 3].z = bottom.z;
    nodes[4 * index + 3].w = top.z;
}
GRACE_HOST_DEVICE void set_right_AABB(float3 bottom, float3 top,
                                      int index, int4* nodes)
{
    set_right_AABB(bottom, top, index, reinterpret_cast<float4*>(nodes));
}
GRACE_DEVICE void set_right_AABB(float3 bottom, float3 top,
                                 int index, volatile int4* nodes)
{
    set_right_AABB(bottom, top, index, reinterpret_cast<volatile float4*>(nodes));
}

} // namespace grace
