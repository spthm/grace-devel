#pragma once

// TODO: C++11, #include <cstdint>
#include <stdint.h>

namespace grace {

#if defined(__GNUC__) || defined(__clang__) || defined(__APPLE__) && defined(__MACH__)
typedef uint32_t uinteger32;
typedef uint64_t uinteger64;
typedef int32_t integer32;
typedef int64_t integer64;
#elif defined(_MSC_VER)
typedef unsigned __int32 uinteger32;
typedef unsigned __int64 uinteger64;
typedef __int32 integer32;
typedef __int64 integer64;
#else
#warning Compiler or system not detected, key bit-lengths may be incorrect
typedef unsigned int uinteger32;
typedef unsigned long long uinteger64;
typedef int integer32;
typedef long long integer64;
#endif

// Binary encoding with +ve = 1, -ve = 0.
// Octants for ray generation.
enum Octants {
    PPP = 7,
    PPM = 6,
    PMP = 5,
    PMM = 4,
    MPP = 3,
    MPM = 2,
    MMP = 1,
    MMM = 0
};

enum RaySortType {
    NoSort,
    DirectionSort,
    EndPointSort
};

// ParallelRays -> Multiple rays are tested, in parallel, against a single
//                 primitive (default).
// ParallelPrimitives -> Multiple primitives are tested, in parallel, against a
//                       single ray. This has some overhead.
struct LeafTraversal {
    enum E {
        ParallelRays,
        ParallelPrimitives
    };
};

} // namespace grace
