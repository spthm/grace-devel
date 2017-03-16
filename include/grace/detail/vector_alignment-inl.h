#pragma once

// No grace/vector.h include.
// This should only ever be included by grace/vector.h.

namespace grace {

namespace detail {

// Default case occurs when Dims > 4 or (Tsize > 8 and Tsize != 16).
// This also catches the non-vector Dims == 1.
// This default case is why we need to pass T as a template argument, even
// though it is not used by any other vector_alignment_ specialization.
template <size_t Dims, typename T, size_t Tsize>
struct vector_alignment_
{
    static const int value = GRACE_ALIGNOF(T);
};

template <size_t Dims, typename T>
struct vector_alignment
{
    static const int value = vector_alignment_<Dims, T, sizeof(T)>::value;
};


// The below template specializations balance the most-efficient alignment
// (i.e. one for which nvcc can emit vector loads) against any resulting
// increase in the size of the Vector type. For sizeof(T) <= 4 and
// sizeof(T) == 8, efficient memory access is favoured. For sizeof(T) == 16,
// both are optimal. For other sizeof(T) values, keeping struct padding to a
// minimum is favoured. The alignments are given by the table below.
//
//                   alignof(Vector<Dims, T>)
//                          sizeof(T)
// Dims   1    2    3    4    5    6    7    8    9+   16
//  1     D    D    D    D    D    D    D    8    D    16
//  2     2    4    8    8    4    4    16   16   D    16
//  3     4    8    4    16   16   4    8    16   D    16
//  4     4    8    16   16   4    8    16   16   D    16
//  5+    D    D    D    D    D    D    D    8    D    16
//
// A 'D' denotes the default specification, i.e. whatever the compiler wants.


//
// Specializations for 8- and 16-byte T
//

// If sizeof(T) == 8, 8 should be our minimum alignment. It wastes no space, and
// allows for efficient vector4 loads on CUDA devices.
template <size_t Dims, typename T>
struct vector_alignment_<Dims, T, 8>
{
    static const int value = 8;
};

// If sizeof(T) == 16, 16 should be our alignment. It wastes no space, and
// allows for efficient vector4 loads on CUDA devices.
template <size_t Dims, typename T>
struct vector_alignment_<Dims, T, 16>
{
    static const int value = 16;
};


//
// 8-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 8>
{
    static const int value = 16;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 8>
{
    static const int value = 16;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 8>
{
    static const int value = 16;
};


//
// 7-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 7>
{
    static const int value = 16;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 7>
{
    static const int value = 8;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 7>
{
    static const int value = 16;
};


//
// 6-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 6>
{
    static const int value = 4;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 6>
{
    static const int value = 4;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 6>
{
    static const int value = 8;
};


//
// 5-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 5>
{
    static const int value = 4;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 5>
{
    static const int value = 16;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 5>
{
    static const int value = 4;
};


//
// 4-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 4>
{
    static const int value = 8;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 4>
{
    static const int value = 16;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 4>
{
    static const int value = 16;
};


//
// 3-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 3>
{
    static const int value = 8;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 3>
{
    static const int value = 4;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 3>
{
    static const int value = 16;
};


//
// 2-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 2>
{
    static const int value = 4;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 2>
{
    static const int value = 8;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 2>
{
    static const int value = 8;
};


//
// 1-byte T
//

// Vector<2, >
template <typename T>
struct vector_alignment_<2, T, 1>
{
    static const int value = 2;
};

// Vector<3, >
template <typename T>
struct vector_alignment_<3, T, 1>
{
    static const int value = 4;
};

// Vector<4, >
template <typename T>
struct vector_alignment_<4, T, 1>
{
    static const int value = 4;
};

} // namespace detail

} // namespace grace
