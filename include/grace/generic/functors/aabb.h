#pragma once

#include "grace/sphere.h"
#include "grace/types.h"
#include "grace/vector.h"

#include <iterator>

namespace grace {

struct AABBSphere
{
    template <typename T>
    GRACE_HOST_DEVICE void operator()(
        Sphere<T> sphere,
        Vector<3, T>* bot,
        Vector<3, T>* top) const
    {
        bot->x = sphere.x - sphere.r;
        top->x = sphere.x + sphere.r;

        bot->y = sphere.y - sphere.r;
        top->y = sphere.y + sphere.r;

        bot->z = sphere.z - sphere.r;
        top->z = sphere.z + sphere.r;
    }
};

namespace detail {

// Finds the centroid of an AABB.
GRACE_HOST_DEVICE Vector<3, float> AABB_centroid(const Vector<3, float> bot,
                                                 const Vector<3, float> top)
{
    Vector<3, float> centre;
    centre.x = (bot.x + top.x) / 2.f;
    centre.y = (bot.y + top.y) / 2.f;
    centre.z = (bot.z + top.z) / 2.f;

    return centre;
}

GRACE_HOST_DEVICE Vector<3, double> AABB_centroid(const Vector<3, double> bot,
                                                  const Vector<3, double> top)
{
    Vector<3, double> centre;
    centre.x = (bot.x + top.x) / 2.;
    centre.y = (bot.y + top.y) / 2.;
    centre.z = (bot.z + top.z) / 2.;

    return centre;
}

} // namespace detail

} // namespace grace
