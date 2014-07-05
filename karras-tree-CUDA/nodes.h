#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace grace {

class Nodes
{
public:
    // A 32-bit int allows for up to indices up to +2,147,483,647 > 1024^3.
    // Since we likely can't fit more than 1024^3 particles on the GPU, int ---
    // which is 32-bit on relevant platforms --- should be sufficient for the
    // indices.
    // We could use uint and set the root node to 1.  Then common_prefix_length
    // may return 0 for out-of-range results, rather than -1.
    //
    // int4.x: left child index.
    //     .y: right child index.
    //     .z: span (such that i + nodes[i].z == the last or first sphere in the
    //         ith node for positive or negative span, respectively.
    //     .w: parent index.
    thrust::device_vector<int4> hierarchy;
    // Equal to the common prefix of the keys which this node spans.
    // Currently used only when verifying correct construction.
    thrust::device_vector<unsigned int> level;

    // AABB[3*node_ID + 0].x = left_bx
    //                    .y = left_tx
    //                    .z = right_bx
    //                    .w = right_tx
    // AABB[3*node_ID + 1].x = left_by
    //                    .y = left_ty
    //                    .z = right_by
    //                    .w = right_ty
    // AABB[3*node_ID + 2].x = left_bz
    //                    .y = left_tz
    //                    .z = right_bz
    //                    .w = right_tz
    thrust::device_vector<float4> AABB;

    Nodes(unsigned int N_nodes) : hierarchy(N_nodes), level(N_nodes),
                                  AABB(3*N_nodes) {}
};

class Leaves
{
public:
    // indices.x = first; indices.y = span; indices.z = parent; indices.w = pad.
    thrust::device_vector<int4> indices;

    Leaves(unsigned int N_leaves) : indices(N_leaves) {}
};

class H_Nodes
{
public:
    thrust::host_vector<int4> hierarchy;
    thrust::host_vector<unsigned int> level;
    thrust::host_vector<float4> AABB;

    H_Nodes(unsigned int N_nodes) : hierarchy(N_nodes), level(N_nodes),
                                    AABB(3*N_nodes) {}
};

class H_Leaves
{
public:
    thrust::host_vector<int4> indices;

    H_Leaves(unsigned int N_leaves) : indices(N_leaves) {}
};

} //namespace grace
