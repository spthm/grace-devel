#pragma once

// No grace/vector.h include.
// This should only ever be included by grace/vector.h.

namespace grace {

namespace detail {

// Default case occurs when Dims > 4 or (Tsize > 8 and Tsize != 16).
// This also catches the non-vector Dims == 1.
template <size_t Dims, size_t Tsize, typename T>
struct vector_base {};


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
GRACE_ALIGNED_STRUCT(8) vector_base<Dims, 8, T> {};

// If sizeof(T) == 16, 16 should be our alignment. It wastes no space, and
// allows for efficient vector4 loads on CUDA devices.
template <size_t Dims, typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<Dims, 16, T> {};


//
// 8-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<2, 8, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<3, 8, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<4, 8, T> {};


//
// 7-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<2, 7, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) vector_base<3, 7, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<4, 7, T> {};


//
// 6-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<2, 6, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<3, 6, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) vector_base<4, 6, T> {};


//
// 5-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<2, 5, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<3, 5, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<4, 5, T> {};


//
// 4-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) vector_base<2, 4, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<3, 4, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<4, 4, T> {};


//
// 3-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) vector_base<2, 3, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<3, 3, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) vector_base<4, 3, T> {};


//
// 2-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<2, 2, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) vector_base<3, 2, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) vector_base<4, 2, T> {};


//
// 1-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(2) vector_base<2, 1, T> {};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<3, 1, T> {};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) vector_base<4, 1, T> {};

} // namespace detail

} // namespace grace
