#pragma once

#include <stdint.h>

#define UInteger32 uint32_t
#define UInteger64 uint64_t

namespace grace {

template <typename T>
struct Vector3 {
    T x, y, z;
};

} // namespace grace
