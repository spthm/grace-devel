namespace grace {

namespace gpu {

template <typename UInteger>
__device__ UInteger space_by_1(UInteger unspaced, int order=10);

template <typename UInteger>
__device__ UInteger space_by_2(UInteger unspaced, int order=10);

template <typename UInteger>
__device__ UInteger bit_prefix(UInteger a, UInteger b);

} // namespace gpu

} // namespace grace
