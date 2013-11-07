#pragma once

#include <stdint.h>

#define UInteger32 uint32_t
#define UInteger64 uint64_t
#define Integer32 int32_t
#define Integer64 int64_t

#define MORTON_THREADS_PER_BLOCK 512
#define BUILD_THREADS_PER_BLOCK 512
#define AABB_THREADS_PER_BLOCK 512
#define MAX_BLOCKS 112 // 7MPs * 16 blocks/MP for compute capability 3.0.

namespace grace {

template <typename T>
struct Vector3 {
    T x, y, z;
    Vector3() : x(0), y(0), z(0) {}
    Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
};

} // namespace grace
