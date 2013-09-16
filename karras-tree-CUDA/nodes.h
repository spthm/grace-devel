namespace grace {

template <typename UInteger>
struct Node
{
    UInteger left;
    UInteger right;
    UInteger parent;

    bool left_leaf_flag;
    bool right_leaf_flag;

    // AABB.
    float top[3];
    float bottom[3];
};

struct Leaf
{
    UInteger parent;

    float top[3];
    float bottom[3];
};

} //namespace grace
