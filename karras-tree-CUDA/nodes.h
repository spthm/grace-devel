#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace grace {

struct Box
{
    float tx, ty, tz;
    float bx, by, bz;
};

class Nodes
{
public:
    // A 32-bit int allows for up to indices up to +2,147,483,647 > 1024^3.
    // Since we likely can't fit more than 1024^3 particles on the GPU, int
    // - which is 32-bit on relevant platforms - should be sufficient for the
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

    thrust::device_vector<Box> AABB;

    Nodes(unsigned int N_nodes) : hierarchy(N_nodes), level(N_nodes),
                                  AABB(N_nodes) {}
};

class Leaves
{
public:
    // indices.x = first; indices.y = span; indices.z = parent; indices.w = pad.
    thrust::device_vector<int4> indices;

    thrust::device_vector<Box> AABB;

    Leaves(unsigned int N_leaves) : indices(N_leaves), AABB(N_leaves) {}
};

class H_Nodes
{
public:
    thrust::host_vector<int4> hierarchy;
    thrust::host_vector<unsigned int> level;
    thrust::host_vector<Box> AABB;

    H_Nodes(unsigned int N_nodes) : hierarchy(N_nodes), level(N_nodes),
                                    AABB(N_nodes) {}
};

class H_Leaves
{
public:
    thrust::host_vector<int4> indices;
    thrust::host_vector<Box> AABB;

    H_Leaves(unsigned int N_leaves) : indices(N_leaves), AABB(N_leaves) {}
};

} //namespace grace
