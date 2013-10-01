#pragma once

#include <stdint.h>

#define UInteger32 uint32_t
#define UInteger64 uint64_t

namespace grace {

template <typename T>
struct Vector3 {
    T x, y, z;
    Vector3() : x(0), y(0), z(0) {}
    Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
};

} // namespace grace
