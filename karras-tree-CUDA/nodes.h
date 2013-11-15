#pragma once

#include "types.h"

namespace grace {

struct Node
{
    // A 32-bit int allows for up to 2,147,483,647 > 1024^3.
    // Since we likely can't fit more than 1024^3 particles on the GPU
    // anyway, int32 should be sufficient for the indices.
    // Could use uint32 and set the root node to 1.  Then common_prefix
    // may return 0 for out-of-range results, rather than -1.
    Integer32 left;
    Integer32 right;
    Integer32 parent;
    Integer32 far_end;

    // Equal to the common prefix of the keys which this node spans.
    unsigned int level;

    bool left_leaf_flag;
    bool right_leaf_flag;

    // AABB.  float should be sufficient (i.e. no need for double).
    float top[3];
    float bottom[3];
};

struct Leaf
{
    Integer32 parent;

    float top[3];
    float bottom[3];
};

} //namespace grace
