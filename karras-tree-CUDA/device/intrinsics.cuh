// See http://hpc.oit.uci.edu/nvidia-doc/sdk-cuda-doc/C/doc/ptx_isa_3.0.pdf
// for details of the scalar vmin/vmax.
// min(min(a, b), c)
__device__ __forceinline__ int min_vmin(int a, int b, int c) {
    int mvm;
    asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// max(max(a, b), c)
__device__ __forceinline__ int max_vmax(int a, int b, int c) {
    int mvm;
    asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// max(min(a, b), c)
__device__ __forceinline__ int max_vmin(int a, int b, int c) {
    int mvm;
    asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// min(max(a, b), c)
__device__ __forceinline__ int min_vmax(int a, int b, int c) {
    int mvm;
    asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}

__device__ __forceinline__ float minf_vminf(float f1, float f2, float f3) {
    return __int_as_float(min_vmin(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __forceinline__ float maxf_vmaxf(float f1, float f2, float f3) {
    return __int_as_float(max_vmax(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __forceinline__ float minf_vmaxf(float f1, float f2, float f3) {
    return __int_as_float(min_vmax(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __forceinline__ float maxf_vminf(float f1, float f2, float f3) {
    return __int_as_float(max_vmin(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
