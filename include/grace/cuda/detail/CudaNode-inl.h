#pragma once

#include "grace/aabb.h"
#include "grace/types.h"
#include "grace/vector.h"

namespace grace {

namespace detail {

// A 32-bit int allows for indices up to 2,147,483,647 > 1024^3.
// Since we likely can't fit more than 1024^3 particles on the GPU, int ---
// which is 32-bit on relevant platforms --- should be sufficient for the
// indices. In ALBVH there is no reason not to use uint, though.
//


GRACE_ALIGNED_STRUCT(16) CudaLeaf
{
private:
    // .x = index of first sphere in this leaf
    // .y = number of spheres in this leaf
    // .z = padding
    // .w = padding
    Vector<4, int> hierarchy;

public:
    GRACE_HOST_DEVICE bool is_inner() const
    {
        // TODO Common node/leaf base class:
        // test hierarchy.w != 0, and have leaves always set hierarchy.w = 0?
        // right-most leaf can never be zero for an inner node.
        return false;
    }

    GRACE_HOST_DEVICE bool is_leaf() const
    {
        return true;
    }

    GRACE_HOST_DEVICE int first_primitive() const
    {
        return hierarchy.x;
    }

    GRACE_HOST_DEVICE void set_first_primitive(const int index)
    {
        hierarchy.x = index;
    }

    GRACE_HOST_DEVICE int last_primitive() const
    {
        return first_primitive() + size() - 1;
    }

    GRACE_HOST_DEVICE int size() const
    {
        return hierarchy.y;
    }

    GRACE_HOST_DEVICE void set_size(const int num_primitives)
    {
        hierarchy.y = num_primitives;
    }
};

GRACE_ALIGNED_STRUCT(16) CudaNode
{
private:
    // .x: left child index
    // .y: right child index
    // .z: index of first leaf in this node
    // .w: index of the last leaf in this node
    Vector<4, int> hierarchy;

    // .x = left.min.x
    // .y = left.max.x
    // .z = left.min.y
    // .w = left.max.y
    Vector<4, float> AABB_Lxy;

    // .x = right.min.x
    // .y = right.max.x
    // .z = right.min.y
    // .w = right.max.y
    Vector<4, float> AABB_Rxy;

    // .x = left.min.z
    // .y = left.max.z
    // .z = right.min.z
    // .w = right.max.z
    Vector<4, float> AABB_LRz;

public:
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
        // Right child can never be at index 0.
        return hierarchy.y > 0;
    }

    GRACE_HOST_DEVICE bool left_is_inner(const size_t N) const
    {
        return hierarchy.x < N;
    }

    GRACE_HOST_DEVICE bool left_is_leaf(const size_t N) const
    {
        return hierarchy.x >= N;
    }

    GRACE_HOST_DEVICE bool right_is_inner(const size_t N) const
    {
        return hierarchy.y < N;
    }

    GRACE_HOST_DEVICE bool right_is_leaf(const size_t N) const
    {
        return hierarchy.y >= N;
    }

    GRACE_HOST_DEVICE int size() const
    {
        return hierarchy.w - hierarcy.z + 1;
    }

    GRACE_HOST_DEVICE AABB<float> left_AABB() const
    {
        const Vector<3, float> min(AABB_Lxy.x, AABB_Lxy.z, AABB_LRz.x);
        const Vector<3, float> max(AABB_Lxy.y, AABB_Lxy.w, AABB_LRz.y);
        return AABB<float>(min, max);
    }

    GRACE_HOST_DEVICE void set_left_AABB(const AABB<float>& aabb)
    {
        AABB_Lxy.x = aabb.min.x;
        AABB_Lxy.z = aabb.min.y;
        AABB_LRz.x = aabb.min.z;

        AABB_Lxy.y = aabb.max.x;
        AABB_Lxy.w = aabb.max.y;
        AABB_LRz.y = aabb.max.z;
    }

    GRACE_HOST_DEVICE AABB<float> right_AABB() const
    {
        const Vector<3, float> min(AABB_Rxy.x, AABB_Rxy.z, AABB_LRz.z);
        const Vector<3, float> max(AABB_Rxy.y, AABB_Rxy.w, AABB_LRz.w);
        return AABB<float>(min, max);
    }

    GRACE_HOST_DEVICE void set_right_AABB(const AABB<float>& aabb)
    {
        AABB_Rxy.x = aabb.min.x;
        AABB_Rxy.z = aabb.min.y;
        AABB_LRz.z = aabb.min.z;

        AABB_Rxy.y = aabb.max.x;
        AABB_Rxy.w = aabb.max.y;
        AABB_LRz.w = aabb.max.z;
    }

    GRACE_HOST_DEVICE int left_child() const
    {
        return hierarchy.x;
    }

    GRACE_HOST_DEVICE void set_left_child(const int index)
    {
        hierarchy.x = index;
    }

    GRACE_HOST_DEVICE int right_child() const
    {
        return hierarchy.y;
    }

    GRACE_HOST_DEVICE void set_right_child(const int index)
    {
        hierarchy.y = index;
    }

    GRACE_HOST_DEVICE int first_leaf() const
    {
        return hierarchy.z;
    }

    GRACE_HOST_DEVICE void set_first_leaf(const int index)
    {
        hierarchy.z = index;
    }

    GRACE_HOST_DEVICE int last_leaf() const
    {
        return hierarchy.w;
    }

    GRACE_HOST_DEVICE void set_last_leaf(const int index)
    {
        hierarchy.w = index;
    }
};

} // namespace detail

} // namespace grace
