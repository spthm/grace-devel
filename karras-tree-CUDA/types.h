#pragma once

#include <stdint.h>

namespace grace {

typedef uint32_t uinteger32;
typedef uint64_t uinteger64;
typedef int32_t integer32;
typedef int64_t integer64;

template <typename T>
struct Vector3 {
    T x, y, z;
    Vector3() : x(0), y(0), z(0) {}
    Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
};

} // namespace grace
