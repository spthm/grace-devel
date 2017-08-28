#pragma once

#include "grace/config.h"

namespace grace {

namespace detail {

// Forward declarations
template <typename T> class Bvh_ref;
template <typename T> class Bvh_const_ref;


template <typename NodeVector, typename LeafVector>
class BvhBase
{
public:
    typedef NodeVector node_vector;
    typedef LeafVector leaf_vector;

    // Allocates space only.
    GRACE_HOST
    explicit BvhBase(const size_t num_primitives, const int max_per_leaf = 1)
        : max_per_leaf_(max_per_leaf), root_index_(-1)
    {
        reserve_nodes(num_primitives);
    }


    GRACE_HOST int max_per_leaf() const
    {
        return max_per_leaf_;
    }
    GRACE_HOST size_t num_nodes() const
    {
        return nodes_.size();
    }
    GRACE_HOST size_t num_leaves() const
    {
        return leaves_.size();
    }
    GRACE_HOST size_t root_index() const
    {
        return root_index_;
    }
    GRACE_HOST void set_root_index(const int index)
    {
        root_index_ = index;
    }

private:
    const int max_per_leaf_;
    int root_index_;
    node_vector nodes_;
    leaf_vector leaves_;

    GRACE_HOST void reserve_nodes(const size_t num_primitives)
    {
        size_t estimate = num_primitives;
        if (max_per_leaf_ > 1) {
            estimate = (size_t)(1.7 * (num_primitives / max_per_leaf_));
        }

        nodes_.reserve(estimate);
        leaves_.reserve(estimate);
    }

    // Bvh_ref will actually be templated over some derived class, but we
    // still need it to be a friend of this class.
    // So friend class BVH[_const]_ref<BvhBase> is insufficient.
    // We could instead template Bvh_ref over Vector, Node and Leaf, but that
    // makes it a little more cumbersome to use, while the increased type
    // safety is of little benefit.
    template <typename U> friend class Bvh_ref;
    template <typename U> friend class Bvh_const_ref;
};

} // namespace detail

} // namespace grace

#include "grace/detail/bvh_ref.h"
