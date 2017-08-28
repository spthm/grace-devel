#pragma once

#include "grace/aabb.h"
#include "grace/config.h"
#include "grace/vector.h"

namespace grace {

namespace detail {


struct BvhLeaf
{
private:
    AABBf aabb_;
    int first_;
    int size_;
    int parent_;

public:
    GRACE_HOST_DEVICE BvhLeaf()
        : aabb_(AABBf()), first_(-1), size_(0), parent_(-1) {}

    GRACE_HOST_DEVICE BvhLeaf(const int first_primitive,
                              const int num_primitives,
                              const int parent,
                              const AABBf aabb)
        : aabb_(aabb),
          first_(first_primitive),
          size_(num_primitives),
          parent_(parent) {}

    GRACE_HOST_DEVICE BvhLeaf(const BvhLeaf& other)
        : aabb_(other.aabb_),
          first_(other.first_),
          size_(other.size_),
          parent_(other.parent_) {}

    GRACE_HOST_DEVICE AABBf aabb() const
    {
        return aabb_;
    }

    GRACE_HOST_DEVICE void set_aabb(const AABBf aabb)
    {
        aabb_ = aabb;
    }

    GRACE_HOST_DEVICE bool is_inner() const
    {
        // TODO Common node/leaf base class.
        return false;
    }

    GRACE_HOST_DEVICE bool is_leaf() const
    {
        return true;
    }

    GRACE_HOST_DEVICE bool is_empty() const
    {
        return size_ == 0;
    }

    GRACE_HOST_DEVICE int first_primitive() const
    {
        return first_;
    }

    GRACE_HOST_DEVICE void set_first_primitive(const int index)
    {
        first_ = index;
    }

    GRACE_HOST_DEVICE int last_primitive() const
    {
        return first_primitive() + size() - 1;
    }

    GRACE_HOST_DEVICE int parent() const
    {
        return parent_;
    }

    GRACE_HOST_DEVICE void set_parent(const int index)
    {
        parent_ = index;
    }

    GRACE_HOST_DEVICE int size() const
    {
        return size_;
    }

    GRACE_HOST_DEVICE void set_size(const int num_primitives)
    {
        size_ = num_primitives;
    }
};

struct BvhNode
{
private:
    AABBf aabb_;
    int left_;
    int right_;
    int first_;
    int size_;

public:
    GRACE_HOST_DEVICE BvhNode()
        : aabb_(AABBf()), left_(-1), right_(-1), first_(-1), size_(0) {}

    GRACE_HOST_DEVICE BvhNode(const int left_child, const int right_child,
                              const int first_leaf, const int size,
                              const AABBf aabb)
        : aabb_(aabb),
          left_(left_child),
          right_(right_child),
          first_(first_leaf),
          size_(size) {}

    GRACE_HOST_DEVICE BvhNode(const BvhNode& other)
        : aabb_(other.aabb_),
          left_(other.left_),
          right_(other.right_),
          first_(other.first_),
          size_(other.size_) {}

    GRACE_HOST_DEVICE bool is_inner() const
    {
        return true;
    }

    GRACE_HOST_DEVICE bool is_leaf() const
    {
        return false;
    }

    GRACE_HOST_DEVICE bool is_empty() const
    {
        return size_ == 0;
    }

    GRACE_HOST_DEVICE bool left_is_inner(const size_t N) const
    {
        return left_ < N;
    }

    GRACE_HOST_DEVICE bool left_is_leaf(const size_t N) const
    {
        return left_ >= N;
    }

    GRACE_HOST_DEVICE bool right_is_inner(const size_t N) const
    {
        return right_ < N;
    }

    GRACE_HOST_DEVICE bool right_is_leaf(const size_t N) const
    {
        return right_ >= N;
    }

    GRACE_HOST_DEVICE AABBf aabb() const
    {
        return aabb_;

    }

    GRACE_HOST_DEVICE void set_aabb(const AABBf& aabb)
    {
        aabb_ = aabb;
    }

    GRACE_HOST_DEVICE int left_child() const
    {
        return left_;
    }

    GRACE_HOST_DEVICE void set_left_child(const int index)
    {
        left_ = index;
    }

    GRACE_HOST_DEVICE int right_child() const
    {
        return right_;
    }

    GRACE_HOST_DEVICE void set_right_child(const int index)
    {
        right_ = index;
    }

    GRACE_HOST_DEVICE int first_leaf() const
    {
        return first_;
    }

    GRACE_HOST_DEVICE void set_first_leaf(const int index)
    {
        first_ = index;
    }

    GRACE_HOST_DEVICE int size() const
    {
        return size_;
    }

    GRACE_HOST_DEVICE void set_size(const int size)
    {
        size_ = size;
    }
};

// TODO: Template this over Node, Leaf types, because the implementation is
//       always the same. Then move it somewhere else.
struct is_empty_bvh_node
{
    GRACE_HOST_DEVICE bool operator()(const BvhLeaf& leaf)
    {
        return leaf.is_empty();
    }

    GRACE_HOST_DEVICE bool operator()(const BvhNode& node)
    {
        return node.is_empty();
    }
};

} // namespace detail

} // namespace grace
