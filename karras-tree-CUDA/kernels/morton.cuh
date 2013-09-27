#pragma once

#include "../types.h"
#include "bits.cuh"

namespace grace {

template <typename UInteger>
__host__ __device__ UInteger32 morton_key_30bit(UInteger x, UInteger y, UInteger z) {
    return space_by_two_10bit(z) << 2 | space_by_two_10bit(y) << 1 | space_by_two_10bit(x);
}

template <typename UInteger>
__host__ __device__ UInteger64 morton_key_63bit(UInteger x, UInteger y, UInteger z) {
    return space_by_two_21bit(z) << 2 | space_by_two_21bit(y) << 1 | space_by_two_21bit(x);
}

template <typename UInteger, typename Float>
class morton_key_functor {};

template <typename Float>
class morton_key_functor<UInteger32, Float>
{
    static const unsigned int span = (1u << 10) - 1;
    const Vector3<Float> scale;

    // span = 2^(order) - 1; order = floor(bits_in_key / 3)
    morton_key_functor(const Vector3<Float> AABB_bottom,
                       const Vector3<Float> AABB_top)
    {
        scale((Float)span / (AABB_top.x - AABB_bottom.x),
              (Float)span / (AABB_top.y - AABB_bottom.y),
              (Float)span / (AABB_top.z - AABB_bottom.z));
    }

    __host__  __device__ UInteger32 operator() (const Vector3<Float> pos) {

        UInteger32 x = (UInteger32) pos.x * scale.x;
        UInteger32 y = (UInteger32) pos.y * scale.y;
        UInteger32 z = (UInteger32) pos.z * scale.z;

        return morton_key_30bit(x, y, z);
    }
};

template <typename Float>
class morton_key_functor<UInteger64, Float>
{
    // span = 2^(order) - 1; order = floor(bits_in_key / 3)
    static const unsigned int span = (1u << 21) - 1;
    const Vector3<Float> scale;

    morton_key_functor(const Vector3<Float> AABB_bottom,
                       const Vector3<Float> AABB_top)
    {
        scale((Float)span / (AABB_top.x - AABB_bottom.x),
              (Float)span / (AABB_top.y - AABB_bottom.y),
              (Float)span / (AABB_top.z - AABB_bottom.z));
    }

    __host__  __device__ UInteger64 operator() (const Vector3<Float> pos) {

        UInteger32 x = (UInteger32) pos.x * scale.x;
        UInteger32 y = (UInteger32) pos.y * scale.y;
        UInteger32 z = (UInteger32) pos.z * scale.z;

        return morton_key_63bit(x, y, z);
    }
};

} // namespace grace
