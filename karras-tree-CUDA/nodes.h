#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "types.h"

namespace grace {

struct Box
{
    float tx, ty, tz;
    float bx, by, bz;
};

class Nodes
{
public:
    thrust::device_vector<integer32> left;
    thrust::device_vector<integer32> right;
    thrust::device_vector<integer32> parent;
    thrust::device_vector<integer32> end;

    thrust::device_vector<unsigned int> level;

    thrust::device_vector<Box> AABB;

    Nodes(unsigned int N_nodes) : left(N_nodes), right(N_nodes),
                                  parent(N_nodes),
                                  end(N_nodes),
                                  level(N_nodes),
                                  AABB(N_nodes) {}
};

class Leaves
{
public:
    thrust::device_vector<integer32> parent;

    thrust::device_vector<Box> AABB;

    Leaves(unsigned int N_leaves) : parent(N_leaves), AABB(N_leaves) {}
};

class H_Nodes
{
public:
    thrust::host_vector<integer32> left;
    thrust::host_vector<integer32> right;
    thrust::host_vector<integer32> parent;
    thrust::host_vector<integer32> end;

    thrust::host_vector<unsigned int> level;

    thrust::host_vector<Box> AABB;

    H_Nodes(unsigned int N_nodes) : left(N_nodes), right(N_nodes),
                                    parent(N_nodes),
                                    end(N_nodes),
                                    level(N_nodes),
                                    AABB(N_nodes) {}
};

class H_Leaves
{
public:
    thrust::host_vector<integer32> parent;

    thrust::host_vector<Box> AABB;

    H_Leaves(unsigned int N_leaves) : parent(N_leaves), AABB(N_leaves) {}
};

struct Node
{
    // A 32-bit int allows for up to 2,147,483,647 > 1024^3.
    // Since we likely can't fit more than 1024^3 particles on the GPU
    // anyway, int32 should be sufficient for the indices.
    // Could use uint32 and set the root node to 1.  Then common_prefix
    // may return 0 for out-of-range results, rather than -1.
    integer32 left;
    integer32 right;
    integer32 parent;
    integer32 far_end;

    // Equal to the common prefix of the keys which this node spans.
    unsigned int level;

    // AABB.  float should be sufficient (i.e. no need for double).
    float top[3];
    float bottom[3];

};

struct Leaf
{
    integer32 parent;

    float top[3];
    float bottom[3];
};

} //namespace grace
