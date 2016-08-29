#pragma once

#include "grace/types.h"

namespace grace {

namespace detail {

// Default case occurs when Dims > 4 or Tsize > 8. Use alignment of underlying T
// T type.
// This also catches the non-vector Dims == 1.
template <size_t Dims, size_t Tsize, typename T>
struct VectorMembers
{
    T array[Dims];

    GRACE_HOST_DEVICE VectorMembers()
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = 0;
    }

    template <typename U>
    GRACE_HOST_DEVICE VectorMembers(U init[Dims])
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = init[i];
    }

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<Dims, Usize, U>& rhs)
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = rhs[i];

        return *this;
    }
};


//
// Specializations for 8- and 16-byte T
//

// If sizeof(T) == 8, 8 should be our minimum alignment. It wastes no space, and
// allows for efficient vector4 loads on CUDA devices.
template <size_t Dims, typename T>
GRACE_ALIGNED_STRUCT(8) VectorMembers<Dims, 8, T>
{
    T array[Dims];

    GRACE_HOST_DEVICE VectorMembers()
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = 0;
    }

    template <typename U>
    GRACE_HOST_DEVICE VectorMembers(U init[Dims])
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = init[i];
    }

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<Dims, Usize, U>& rhs)
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = rhs[i];

        return *this;
    }
};

// If sizeof(T) == 16, 16 should be our alignment. It wastes no space, and
// allows for efficient vector4 loads on CUDA devices.
template <size_t Dims, typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<Dims, 16, T>
{
    T array[Dims];

    GRACE_HOST_DEVICE VectorMembers()
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = 0;
    }

    template <typename U>
    GRACE_HOST_DEVICE VectorMembers(U init[Dims])
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = init[i];
    }

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<Dims, Usize, U>& rhs)
    {
        for (size_t i = 0; i < Dims; ++i)
            array[i] = rhs[i];

        return *this;
    }
};


//
// Vector<{2, 3, 4}, T> partial specializations.
//

// Vector<2, T> always has x, y members, but for arbitrary T array-accessor is
// unsafe!
template <size_t Tsize, typename T>
struct VectorMembers<2, Tsize, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }
};

// Vector<3, > always has x, y and z members, but for arbitrary T array-accessor
// is unsafe.
template <size_t Tsize, typename T>
struct VectorMembers<3, Tsize, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }
};

// Vector<4, > always has x, y, z and w members, but for arbitrary T
// array-accessor is unsafe.
template <size_t Tsize, typename T>
struct VectorMembers<4, Tsize, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};


// The below template specializations balance the most-efficient alignment
// (i.e. one for which nvcc can emit vector loads) against any resulting
// increase in the size of the Vector type. For sizeof(T) <= 4 and
// sizeof(T) == 8, efficient memory access is favoured. For sizeof(T) == 16,
// both are optimal. For other sizeof(T) values, keeping struct padding to a
// minimum is favoured. The alignments are given by the table below.
//
// Further, padding is explicitly added to increase safety when accessing
// members as an array/pointer. GRACE_ALIGNED_STRUCT is thus not strictly
// needed, but is left for clarity of intent. Explicit padding is also
// identified in the table below.
//
//               alignof(Vector<Dims, T>) (explicitly padded y/n)
//                                   sizeof(T)
// Dims    1     2       3        4      5      6       7       8      9+    16
//  1      x     x       x        x      x      x       x      8 (n)   x   16 (n)
//  2    2 (n)  4 (n)   8 (y)   8 (n)   4 (y)  4 (n)  16 (y)  16 (n)   x   16 (n)
//  3    4 (y)  8 (y)   4 (y)  16 (y)  16 (y)  4 (y)   8 (y)  16 (y)   x   16 (n)
//  4    4 (n)  8 (n)  16 (y)  16 (n)   4 (n)  8 (n)  16 (y)  16 (n)   x   16 (n)
//  5+     x     x       x        x      x      x       x      8 (n)   x   16 (n)
//
// An 'x' denotes the default specification and no explicit padding, i.e.
// whatever the compiler wants.
//
// Also note that the below specializations for Dims = {2, 3, 4} combined with
// sizeof(T) = {8, 16} are necessary to resolve the ambiguity in the
// separate Dims = {2, 3, 4} and sizeof(T) = {8, 16} partial specializations
// above.


//
// 16-byte T
//

// Vector<2, >
// Required to resolve ambiguity when Dims == 2 and sizeof(T) == 16.
// No explicit padding. No padding added by compiler.
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<2, 16, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }
};

// Vector<3, >
// Required to resolve ambiguity when Dims == 3 and sizeof(T) == 16.
// No explicit padding. No padding added by compiler.
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<3, 16, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }
};

// Vector<4, >
// Required to resolve ambiguity when Dims == 4 and sizeof(T) == 16.
// No explicit padding. No padding added by compiler.
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<4, 16, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};


//
// 8-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<2, 8, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<3, 8, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding[8]; // Prevent compiler from inserting padding between public members.
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<4, 8, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};


//
// 7-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<2, 7, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

private:
    char padding[2];
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) VectorMembers<3, 7, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding[3];
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<4, 7, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }

private:
    char padding[4];
};


//
// 6-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<2, 6, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<3, 6, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding[2];
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) VectorMembers<4, 6, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};


//
// 5-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<2, 5, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

private:
    char padding[2];
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<3, 5, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding[1];
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<4, 5, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};


//
// 4-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) VectorMembers<2, 4, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<3, 4, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding[4]; // Prevent compiler from inserting padding between public members.
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<4, 4, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};


//
// 3-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) VectorMembers<2, 3, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

private:
    char padding[2]; // Prevent compiler from inserting padding between public members.
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<3, 3, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding[3]; // Prevent compiler from inserting padding between public members.
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(16) VectorMembers<4, 3, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }

private:
    char padding[4]; // Prevent compiler from inserting padding between public members.
};


//
// 2-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<2, 2, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) VectorMembers<3, 2, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding[2]; // Prevent compiler from inserting padding between public members.
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(8) VectorMembers<4, 2, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};


//
// 1-byte T
//

// Vector<2, >
template <typename T>
GRACE_ALIGNED_STRUCT(2) VectorMembers<2, 1, T>
{
    T x;
    T y;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y) : x(x), y(y) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<2, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }
};

// Vector<3, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<3, 1, T>
{
    T x;
    T y;
    T z;

    GRACE_HOST_DEVICE VectorMembers() : x(0), y(0), z(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<3, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

private:
    char padding; // Prevent compiler from inserting padding between public members.
};

// Vector<4, >
template <typename T>
GRACE_ALIGNED_STRUCT(4) VectorMembers<4, 1, T>
{
    T x;
    T y;
    T z;
    T w;

    GRACE_HOST_DEVICE VectorMembers() :
        x(0), y(0), z(0), w(0) {}
    GRACE_HOST_DEVICE VectorMembers(const T x, const T y, const T z, const T w)
        : x(x), y(y), z(z), w(w) {}

    template <size_t Usize, typename U>
    GRACE_HOST_DEVICE VectorMembers& operator=(
        const VectorMembers<4, Usize, U>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }
};

} // namespace detail

} // namespace grace
