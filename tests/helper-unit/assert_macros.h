#pragma once

// To avoid screwing with the definition of assert(), we don't want to include
// anything before assert.h.

#ifdef NDEBUG
#define REDEFINE_NDEBUG
#endif

#undef NDEBUG
#include <assert.h>

#ifdef REDEFINE_NDEBUG
#define NDEBUG
#endif

#include "grace/config.h"

#include <cstdio>


template <typename T>
GRACE_HOST_DEVICE void assert_equal(T a, T b, const char* file, const int line)
{
    if (a != b) {
        printf("FAILED EQUAL: %s @ %d\n", file, line);
    }

    assert(a == b);
}

template <typename T>
GRACE_HOST_DEVICE void assert_not_equal(T a, T b,
                                        const char* file, const int line)
{
    if (a == b) {
        printf("FAILED NOT EQUAL: %s @ %d\n", file, line);
    }

    assert(a != b);
}

template <typename T>
GRACE_HOST_DEVICE void assert_equal_ptr(T* p1, T* p2,
                                        const char* file, const int line)
{
    if (p1 != p2) {
        printf("FAILED POINTERS EQUAL: %s @ %d\n", file, line);
    }

    assert(p1 == p2);
}

template <typename T>
GRACE_HOST_DEVICE void assert_not_equal_ptr(T* p1, T* p2,
                                            const char* file, const int line)
{
    if (p1 == p2) {
        printf("FAILED POINTERS NOT EQUAL: %s @ %d\n", file, line);
    }

    assert(p1 != p2);
}

template <typename T>
GRACE_HOST_DEVICE void assert_less_than(T a, T b,
                                        const char* file, const int line)
{
    if (!(a < b)) {
        printf("FAILED LHS < RHS: %s @ %d\n", file, line);
    }

    assert(a < b);
}

template <typename T>
GRACE_HOST_DEVICE void assert_less_than_equal(T a, T b,
                                              const char* file, const int line)
{
    if (!(a <= b)) {
        printf("FAILED LHS <= RHS: %s @ %d\n", file, line);
    }

    assert(a <= b);
}

template <typename T>
GRACE_HOST_DEVICE void assert_zero(T a, const char* file, const int line)
{
    if (a != T(0)) {
        printf("FAILED EQUAL 0: %s @ %d\n", file, line);
    }

    assert(a == T(0));
}

#define ASSERT_EQUAL(a, b) { assert_equal((a), (b), __FILE__, __LINE__); }

#define ASSERT_NOT_EQUAL(a, b) { assert_not_equal((a), (b), __FILE__, __LINE__); }

#define ASSERT_EQUAL_PTR(p1, p2) { assert_equal_ptr((p1), (p2), __FILE__, __LINE__); }

#define ASSERT_NOT_EQUAL_PTR(p1, p2) { assert_not_equal_ptr((p1), (p2), __FILE__, __LINE__); }

#define ASSERT_LESS_THAN(a, b) { assert_less_than((a), (b), __FILE__, __LINE__); }

#define ASSERT_LESS_THAN_EQUAL(a, b) { assert_less_than_equal((a), (b), __FILE__, __LINE__); }

#define ASSERT_ZERO(a) { assert_zero((a), __FILE__, __LINE__); }
