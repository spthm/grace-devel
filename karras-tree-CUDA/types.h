#pragma once

#include <stdint.h>

#define UInteger32 uint32_t
#define UInteger32 uint64_t

template <typename T>
struct Vector3 {
    T x, y, z;
};

