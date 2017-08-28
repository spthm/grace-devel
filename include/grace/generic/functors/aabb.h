#pragma once

#include "grace/config.h"
#include "grace/aabb.h"
#include "grace/sphere.h"
#include "grace/vector.h"

#include <iterator>

namespace grace {

struct AABBSphere
{
    template <typename T>
    GRACE_HOST_DEVICE void operator()(
        Sphere<T> sphere,
        AABB<T>* aabb) const
    {
        aabb->min = sphere.center() - sphere.r;
        aabb->max = sphere.center() + sphere.r;
    }
};

} // namespace grace
