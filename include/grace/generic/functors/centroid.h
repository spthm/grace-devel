#pragma once

#include "grace/sphere.h"
#include "grace/types.h"
#include "grace/vector.h"
#include "grace/generic/functors/aabb.h"

// CUDA
#include <vector_functions.h>

#include <functional>

namespace grace {

// Functor to convert from TPrimitive to a Vector<3, PrecisionType> (i.e. the
// primitive centroid), taking the primitive's centroid to be the centroid of
// the primitive's AABB. AABBFunc must be declared __host__ __device__.
// PrecisionType determines the precision of the AABB and centroid computations.
template <typename PrecisionType, typename TPrimitive, typename AABBFunc>
struct PrimitiveCentroid : public std::unary_function<const TPrimitive&, Vector<3, PrecisionType> >
{
public:
    GRACE_HOST_DEVICE PrimitiveCentroid() : aabb(AABBFunc()) {}

    GRACE_HOST_DEVICE PrimitiveCentroid(AABBFunc aabb) : aabb(aabb) {}

    GRACE_HOST_DEVICE Vector<3, PrecisionType> operator()(const TPrimitive& primitive)
    {
        Vector<3, PrecisionType> bot, top;
        aabb(primitive, &bot, &top);
        return detail::AABB_centroid(bot, top);
    }

private:
    const AABBFunc aabb;
};

template <typename T>
struct CentroidSphere : public std::unary_function<const Sphere<T>&, Vector<3, T> >
{
    GRACE_HOST_DEVICE Vector<3, T> operator()(const Sphere<T>& sphere) const
    {
        return sphere.center();
    }
};

// Useful for any primitive which has .x, .y and .z data members.
// Returns a Vector<3, OutType> instance.
template <typename InType, typename OutType>
struct CentroidPassThrough : public std::unary_function<const InType&, Vector<3, OutType> >
{
    GRACE_HOST_DEVICE Vector<3, OutType> operator()(const InType& primitive) const
    {
        return Vector<3, OutType>(primitive.x, primitive.y, primitive.z);
    }
};

} // namespace grace
