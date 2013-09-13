#include "Nodes.h"

namespace grace {

struct Node
{
    unsigned int left;
    unsigned int right;
    unsigned int parent;

    bool left_leaf_flag;
    bool right_leaf_flag;

    // AABB.
    float top[3];
    float bottom[3];
};

struct Leaf
{
    unsigned int parent;

    float top[3];
    float bottom[3];
};

} //namespace grace
