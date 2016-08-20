#pragma once

#include "grace/point.h"
#include "grace/sphere.h"
#include "grace/types.h"
#include "grace/generic/functors/aabb.h"

// CUDA
#include <vector_functions.h>

#include <functional>

namespace grace {

// Functor to convert from TPrimitive to a Point<float> (primitive centroid),
// taking the primitive's centroid to be the centroid of the primitive's AABB.
// AABBFunc must be declared __host__ __device__, a Thrust requirement.
template <typename TPrimitive, typename AABBFunc>
struct PrimitiveCentroid : public std::unary_function<const TPrimitive&, Point<float> >
{
public:
    GRACE_HOST_DEVICE PrimitiveCentroid() : aabb(AABBFunc()) {}

    GRACE_HOST_DEVICE PrimitiveCentroid(AABBFunc aabb) : aabb(aabb);

    GRACE_HOST_DEVICE Point<float> operator()(const TPrimitive& primitive)
    {
        float3 bot, top;
        aabb(primitive, &bot, &top);
        return detail::AABB_centroid(bot, top);
    }

private:
    const AABBFunc aabb;
};

template <typename T>
struct CentroidSphere : public std::unary_function<const Sphere<T>&, Point<T> >
{
    GRACE_HOST_DEVICE Point<T> operator()(const Sphere<T>& sphere) const
    {
        return Point<T>(sphere);
    }
};

} // namespace grace
