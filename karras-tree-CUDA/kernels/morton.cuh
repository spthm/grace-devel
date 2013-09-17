namepspace grace {

namepspace gpu {

template <typename UInteger, typename Float>
__device__ UInteger float_to_int(Float value, int order=10);

} // namespace gpu

} // namespace grace
