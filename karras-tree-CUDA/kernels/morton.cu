namespace grace {

namespace gpu {

template <typename UInteger, typename Float>
struct morton_key_functor {};

template <typename Float>
struct morton_key_functor<UInteger32, Float>
{
    const UInteger32 span;
    const Vector3<Float> scale;

    // span = 2^(order) - 1; order = floor(bits_in_key / 3)
    morton_key_functor(Vector3<Float> AABB_bottom, Vector3<Float> AABB_top) :
        span((1u << 10) - 1)
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
struct morton_key_functor<UInteger64, Float>
{
    const UInteger32 span;
    const Vector3<Float> scale;

    // span = 2^(order-1); order = floor(bits_in_key / 3)
    morton_key_functor(Vector3<Float> AABB_bottom, Vector3<Float> AABB_top) :
        span((1u << 21) - 1)
    {
        scale((Float)span / (AABB_top.x - AABB_bottom.x),
              (Float)span / (AABB_top.y - AABB_bottom.y),
              (Float)span / (AABB_top.z - AABB_bottom.z));
    }

    __host__  __device__ UInteger64 operator() (const Vector3<Float> pos) {

        UInteger32 x = (UInteger32) pos.x * scale.x;
        UInteger32 y = (UInteger32) pos.y * sclae.y;
        UInteger32 z = (UInteger32) pos.z * scale.z;

        return morton_key_63(x, y, z);
    }
};

__host__ __device__ UInteger32 morton_key_30(UInteger32 x, UInteger32 y, UInteger32 z) {
    return spaced_by_2(z&1023u) << 2 | spaced_by_2(y&1023u) << 1 | spaced_by_2(x&1023u);
}

// TODO: Fix this.  Currently it's actually morton_key_60.
// Use templates on space_by_x and covert x/y/z to UInteger64 in
// morton_key_functor?
__host__ __device__ UInteger64 morton_key_63(UInteger32 x, UInteger32 y, UInteger32 z) {
    return((UInteger64)morton_key_30(x, y, z) |
           (UInteger64)morton_key_30(x >> 10, y >> 10, z >> 10) << 30);
}


} // namespace gpu

} // namespace grace
