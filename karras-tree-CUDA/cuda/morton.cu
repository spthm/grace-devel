namespace grace {

namespace gpu {

template <typename UInteger, typename Float>
struct morton_key_functor
{
    const unsigned int order;
    const UInteger span;
    const Vector3<Float> AABB_inv;

    // order = floor(bits_in_key / 3)
    morton_key_functor(Vector3<Float> AABB_bottom, Vector3<Float> AABB_top) :
        order((unsigned int) CHAR_BIT * sizeof(UInterger) / 3),
        span(1u << order - 1)
    {
        AABB_inv((Float) 1.0 / (AABB_top.x - AABB_bottom.x),
                 (Float) 1.0 / (AABB_top.y - AABB_bottom.y),
                 (Float) 1.0 / (AABB_top.z - AABB_bottom.z));
    }

    __host__  __device__ Vector3<UInteger> operator() (const Vector3<Float> pos) {

        UInteger x = (UInteger) pos.x * AABB_inv.x * span;
        UInteger y = (UInteger) pos.y * AABB_inv.y * span;
        UInteger z = (UInteger) pos.z * AABB_inv.z * span;

        morton_key(x, y, z, order);
    }
};

template <typename UInteger>
morton_key(UInteger x, UInteger y, UInteger z, unsigned int order) {
    return spaced_by_2(z, order) << 2 | spaced_by_2(y, order) << 1 | spaced_by_2(x, order);
}


} // namespace gpu

} // namespace grace
