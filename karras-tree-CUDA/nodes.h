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
    // A 32-bit int allows for up to 2,147,483,647 > 1024^3.
    // Since we likely can't fit more than 1024^3 particles on the GPU
    // anyway, int32 should be sufficient for the indices.
    // Could use uint32 and set the root node to 1.  Then common_prefix
    // may return 0 for out-of-range results, rather than -1.
    //
    // Left, right, parent and end indices.
    thrust::device_vector<int4> lrpe;
    // Equal to the common prefix of the keys which this node spans.
    // Currently used only when verifying correct construction.
    thrust::device_vector<unsigned int> level;

    thrust::device_vector<Box> AABB;

    Nodes(unsigned int N_nodes) : lrpe(N_nodes), level(N_nodes),
                                  AABB(N_nodes) {}
};

class Leaves
{
public:
    thrust::device_vector<int> parent;

    thrust::device_vector<Box> AABB;

    Leaves(unsigned int N_leaves) : parent(N_leaves), AABB(N_leaves) {}
};

class H_Nodes
{
public:
    thrust::host_vector<int4> lrpe;
    thrust::host_vector<unsigned int> level;
    thrust::host_vector<Box> AABB;

    H_Nodes(unsigned int N_nodes) : lrpe(N_nodes), level(N_nodes),
                                    AABB(N_nodes) {}
};

class H_Leaves
{
public:
    thrust::host_vector<int> parent;

    thrust::host_vector<Box> AABB;

    H_Leaves(unsigned int N_leaves) : parent(N_leaves), AABB(N_leaves) {}
};

} //namespace grace
