template <typename UInteger>
__host__ __device__ UInteger space_by_1(UInteger unspaced,
                                        int order) {
    UInteger spaced = 0;
    for (int n=0; n<order; n++) {
        spaced |= ( (unspaced & (1 << n)) << n);
    }
    return spaced;
}

template <typename UInteger>
__host__ __device__ UInteger space_by_2(UInteger unspaced,
                                        int order) {
    UInteger spaced = 0;
    for (int n=0; n<order; n++) {
        spaced |= ( (unspaced & (1 << n)) << 2*n);
    }
    return spaced;
}

template <typename UInteger>
__device__ UInteger bit_prefix(UInteger a, UInteger b) {
    return __clz(a^b);
}

// TODO: Move this somewhere else.
template <typename UInteger>
__host__ __device__ UInteger map_to_int(float value, int order) {
    UInteger span = (1u << order) - 1;
    return (UInteger) value * span;
}
