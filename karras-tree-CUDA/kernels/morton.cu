#include "../types.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

namespace gpu {

template <typename Float>
class morton_key_functor<UInteger32, Float>
{
    const unsigned int span = (1u << 10) - 1;
    const Vector3<Float> scale;

    // span = 2^(order) - 1; order = floor(bits_in_key / 3)
    morton_key_functor(Vector3<Float> AABB_bottom, Vector3<Float> AABB_top)
    {
        scale((Float)span / (AABB_top.x - AABB_bottom.x),
              (Float)span / (AABB_top.y - AABB_bottom.y),
              (Float)span / (AABB_top.z - AABB_bottom.z));
    }

    __host__  __device__ UInteger32 operator() (const Vector3<Float> pos) {

        UInteger32 x = (UInteger32) pos.x * scale.x;
        UInteger32 y = (UInteger32) pos.y * sclae.y;
        UInteger32 z = (UInteger32) pos.z * scale.z;

        return morton_key_30(x, y, z);
    }
};

template <typename Float>
class morton_key_functor<UInteger64, Float>
{
    // span = 2^(order) - 1; order = floor(bits_in_key / 3)
    const unsigned int span = (1u << 21) - 1;
    const Vector3<Float> scale;

    morton_key_functor(Vector3<Float> AABB_bottom, Vector3<Float> AABB_top)
    {
        scale((Float)span / (AABB_top.x - AABB_bottom.x),
              (Float)span / (AABB_top.y - AABB_bottom.y),
              (Float)span / (AABB_top.z - AABB_bottom.z));
    }

    __host__  __device__ UInteger64 operator() (const Vector3<Float> pos) {

        UInteger32 x = (UInteger32) pos.x * scale.x;
        UInteger32 y = (UInteger32) pos.y * scale.y;
        UInteger32 z = (UInteger32) pos.z * scale.z;

        return morton_key_63(x, y, z);
    }
};

__host__ __device__ UInteger32 morton_key_30bit(UInteger32 x, UInteger32 y, UInteger32 z) {
    return space_by_two_10bit(z) << 2 | space_by_two_10bit(y) << 1 | space_by_two_10bit(x);
}

__host__ __device__ UInteger64 morton_key_63bit(UInteger32 x, UInteger32 y, UInteger32 z) {
    return space_by_two_21bit(z) << 2 | space_by_two_21bit(y) << 1 | space_by_two_21bit(x);
}

// Explicitly instantiate the morton_key_functor templates for these
// parameter types only.
template class morton_key_functor<UInteger32, float>;
template class morton_key_functor<UInteger32, double>;
template class morton_key_functor<UInteger32, long double>;
template class morton_key_functor<UInteger64, float>;
template class morton_key_functor<UInteger64, double>;
template class morton_key_functor<UInteger64, long double>;

} // namespace gpu

} // namespace grace
