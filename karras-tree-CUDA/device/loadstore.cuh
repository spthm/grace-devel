#pragma once

/* Load/store functions to be used when specific memory behaviour is required,
 * e.g. a read/write directly from/to L2 cache, which is globally coherent.
 *
 * All take a pointer with the type of the base primitive (i.e. int* for int2
 * read/writes, float* for float4 read/writes) for flexibility.
 *
 * The "memory" clobber is added to all load/store PTX instructions to prevent
 * memory optimizations around the asm statements. We should only be using these
 * functions when we know better than the compiler!
 */

namespace grace {
namespace gpu {

__device__ __forceinline__ int2 load_vec2s32(const int* const addr)
{
    int2 i2;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.ca.v2.s32 {%0, %1}, [%2];" : "=r"(i2.x), "=r"(i2.y) : "l"(addr) : "memory");
    #else
    asm("ld.global.ca.v2.s32 {%0, %1}, [%2];" : "=r"(i2.x), "=r"(i2.y) : "r"(addr) : "memory");
    #endif

    return i2;
}

__device__ __forceinline__ int4 load_vec4s32(const int* const addr)
{
    int4 i4;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.ca.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(i4.x), "=r"(i4.y), "=r"(i4.z), "=r"(i4.w) : "l"(addr) : "memory");
    #else
    asm("ld.global.ca.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(i4.x), "=r"(i4.y), "=r"(i4.z), "=r"(i4.w) : "r"(addr) : "memory");
    #endif

    return i4;
}

__device__ __forceinline__ float4 load_vec4f32(const float* const addr)
{
    float4 f4;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "l"(addr) : "memory");
    #else
    asm("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "r"(addr) : "memory");
    #endif

    return f4;
}

__device__ __forceinline__ float4 load_L2_vec4f32(const float* const addr)
{
    float4 f4;

    #if defined(__LP64__) || defined(_WIN64)
    asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "l"(addr) : "memory");
    #else
    asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(f4.x), "=f"(f4.y), "=f"(f4.z), "=f"(f4.w) : "r"(addr) : "memory");
    #endif

    return f4;
}

// Stores have no output operands, so we additionally mark them as volatile to
// ensure they are not moved or deleted.
__device__ __forceinline__ void store_s32(const int* const addr, const int a)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.s32 [%0], %1;" :: "l"(addr), "r"(a) : "memory");
    #else
    asm volatile ("st.global.wb.s32 [%0], %1;" :: "r"(addr), "r"(a) : "memory");
    #endif
}

__device__ __forceinline__ void store_vec2s32(
    const int* const addr,
    const int a, const int b)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.v2.s32 [%0], {%1, %2};" :: "l"(addr), "r"(a), "r"(b) : "memory");
    #else
    asm volatile ("st.global.wb.v2.s32 [%0], {%1, %2};" :: "r"(addr), "r"(a), "r"(b) : "memory");
    #endif
}

__device__ __forceinline__ void store_vec2f32(
    const float* const addr,
    const float a, const float b)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.v2.f32 [%0], {%1, %2};" :: "l"(addr), "f"(a), "f"(b) : "memory");
    #else
    asm volatile ("st.global.wb.v2.f32 [%0], {%1, %2};" :: "r"(addr), "f"(a), "f"(b) : "memory");
    #endif
}

__device__ __forceinline__ void store_vec4f32(
    const float* const addr,
    const float a, const float b, const float c, const float d)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #else
    asm volatile ("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #endif
}

__device__ __forceinline__ void store_L2_vec2f32(
    const float* const addr,
    const float a, const float b)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.cg.v2.f32 [%0], {%1, %2};" :: "l"(addr), "f"(a), "f"(b) : "memory");
    #else
    asm volatile ("st.global.cg.v2.f32 [%0], {%1, %2};" :: "r"(addr), "f"(a), "f"(b) : "memory");
    #endif
}

__device__ __forceinline__ void store_L2_vec4f32(
    const float* const addr,
    const float a, const float b, const float c, const float d)
{
    #if defined(__LP64__) || defined(_WIN64)
    asm volatile ("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #else
    asm volatile ("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    #endif
}

} // namespace gpu
} // namespace grace
