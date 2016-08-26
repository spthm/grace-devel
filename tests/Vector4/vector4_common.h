#pragma once

// This should be first so assert() is correctly defined.
#include "helper-unit/assert_macros.h"
#include "helper-unit/test_types.h"

#include "grace/types.h"
#include "grace/vector.h"


#include <cstdlib>

#define BLOAT_FACTOR_NONE 1.0
#define BLOAT_FACTOR_SMALL 1.2
#define BLOAT_FACTOR_LARGE 1.5
#define NUM_TEST_VECTORS 9

typedef grace::Vector<4, Byte1T> vec4_1T;
typedef grace::Vector<4, Byte2T> vec4_2T;
typedef grace::Vector<4, Byte3T> vec4_3T;
typedef grace::Vector<4, Byte4T> vec4_4T;
typedef grace::Vector<4, Byte5T> vec4_5T;
typedef grace::Vector<4, Byte6T> vec4_6T;
typedef grace::Vector<4, Byte7T> vec4_7T;
typedef grace::Vector<4, Byte8T> vec4_8T;
typedef grace::Vector<4, Byte16T> vec4_16T;

struct Vector4Ptrs
{
    grace::Vector<4, Byte1T>* vec_1b;
    grace::Vector<4, Byte2T>* vec_2b;
    grace::Vector<4, Byte3T>* vec_3b;
    grace::Vector<4, Byte4T>* vec_4b;
    grace::Vector<4, Byte5T>* vec_5b;
    grace::Vector<4, Byte6T>* vec_6b;
    grace::Vector<4, Byte7T>* vec_7b;
    grace::Vector<4, Byte8T>* vec_8b;
    grace::Vector<4, Byte16T>* vec_16b;
    size_t n;
};

template <typename T>
GRACE_HOST_DEVICE double bloat(const grace::Vector<4, T>& vec)
{
    return sizeof(vec) / static_cast<double>((4 * sizeof(T)));
}

template <typename T>
GRACE_HOST_DEVICE int next_lowest_multiple(const grace::Vector<4, T>& vec)
{
    return (int)(GRACE_ALIGNOF(vec)) * ((int)(sizeof(vec) / GRACE_ALIGNOF(vec)) - 1);
}

GRACE_HOST_DEVICE void set_vector4_data(const Vector4Ptrs vec_ptrs,
                                        const int index, const size_t n_total)
{
    int init = 4 * n_total * index;

    *(vec_ptrs.vec_1b  + index) = vec4_1T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_2b  + index) = vec4_2T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_3b  + index) = vec4_3T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_4b  + index) = vec4_4T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_5b  + index) = vec4_5T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_6b  + index) = vec4_6T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_7b  + index) = vec4_7T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_8b  + index) = vec4_8T( init+1, init+2, init+3, init+4);
    init += 4;
    *(vec_ptrs.vec_16b + index) = vec4_16T(init+1, init+2, init+3, init+4);
}

GRACE_HOST_DEVICE void test_vector4_size_impl(
    const vec4_1T  vec_1b,
    const vec4_2T  vec_2b,
    const vec4_3T  vec_3b,
    const vec4_4T  vec_4b,
    const vec4_5T  vec_5b,
    const vec4_6T  vec_6b,
    const vec4_7T  vec_7b,
    const vec4_8T  vec_8b,
    const vec4_16T vec_16b)
{
    ASSERT_LESS_THAN_EQUAL(bloat(vec_1b),  BLOAT_FACTOR_NONE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_2b),  BLOAT_FACTOR_NONE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_4b),  BLOAT_FACTOR_NONE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_5b),  BLOAT_FACTOR_NONE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_6b),  BLOAT_FACTOR_NONE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_8b),  BLOAT_FACTOR_NONE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_16b), BLOAT_FACTOR_NONE);

    ASSERT_LESS_THAN_EQUAL(bloat(vec_7b),  BLOAT_FACTOR_SMALL);

    ASSERT_LESS_THAN_EQUAL(bloat(vec_3b),  BLOAT_FACTOR_LARGE);

}

GRACE_HOST_DEVICE void test_vector4_size()
{
    // Create on stack.
    test_vector4_size_impl(
        vec4_1T(),
        vec4_2T(),
        vec4_3T(),
        vec4_4T(),
        vec4_5T(),
        vec4_6T(),
        vec4_7T(),
        vec4_8T(),
        vec4_16T()
    );
}

GRACE_HOST_DEVICE void test_vector4_padding_impl(
    const vec4_1T& vec_1b,
    const vec4_2T& vec_2b,
    const vec4_3T& vec_3b,
    const vec4_4T& vec_4b,
    const vec4_5T& vec_5b,
    const vec4_6T& vec_6b,
    const vec4_7T& vec_7b,
    const vec4_8T& vec_8b,
    const vec4_16T& vec_16b)
{
    // sizeof(vec) should be the smallest-possible multiple of alignof(vec).

    ASSERT_ZERO(sizeof(vec_1b)  % GRACE_ALIGNOF(vec_1b));
    ASSERT_ZERO(sizeof(vec_2b)  % GRACE_ALIGNOF(vec_2b));
    ASSERT_ZERO(sizeof(vec_3b)  % GRACE_ALIGNOF(vec_3b));
    ASSERT_ZERO(sizeof(vec_4b)  % GRACE_ALIGNOF(vec_4b));
    ASSERT_ZERO(sizeof(vec_5b)  % GRACE_ALIGNOF(vec_5b));
    ASSERT_ZERO(sizeof(vec_6b)  % GRACE_ALIGNOF(vec_6b));
    ASSERT_ZERO(sizeof(vec_7b)  % GRACE_ALIGNOF(vec_7b));
    ASSERT_ZERO(sizeof(vec_8b)  % GRACE_ALIGNOF(vec_8b));
    ASSERT_ZERO(sizeof(vec_16b) % GRACE_ALIGNOF(vec_16b));

    ASSERT_LESS_THAN(next_lowest_multiple(vec_1b),  (int)(4 * sizeof(Byte1T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_2b),  (int)(4 * sizeof(Byte2T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_3b),  (int)(4 * sizeof(Byte3T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_4b),  (int)(4 * sizeof(Byte4T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_5b),  (int)(4 * sizeof(Byte5T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_6b),  (int)(4 * sizeof(Byte6T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_7b),  (int)(4 * sizeof(Byte7T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_8b),  (int)(4 * sizeof(Byte8T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_16b), (int)(4 * sizeof(Byte16T)));
}

GRACE_HOST_DEVICE void test_vector4_padding()
{
    // Create on stack.
    test_vector4_padding_impl(
        vec4_1T(),
        vec4_2T(),
        vec4_3T(),
        vec4_4T(),
        vec4_5T(),
        vec4_6T(),
        vec4_7T(),
        vec4_8T(),
        vec4_16T()
    );
}

GRACE_HOST_DEVICE void test_vector4_accessors_impl(
    const vec4_1T& vec_1b,
    const vec4_2T& vec_2b,
    const vec4_3T& vec_3b,
    const vec4_4T& vec_4b,
    const vec4_5T& vec_5b,
    const vec4_6T& vec_6b,
    const vec4_7T& vec_7b,
    const vec4_8T& vec_8b,
    const vec4_16T& vec_16b)
{
    ASSERT_EQUAL(vec_1b.x,  vec_1b[ 0]);
    ASSERT_EQUAL(vec_2b.x,  vec_2b[ 0]);
    ASSERT_EQUAL(vec_3b.x,  vec_3b[ 0]);
    ASSERT_EQUAL(vec_4b.x,  vec_4b[ 0]);
    ASSERT_EQUAL(vec_5b.x,  vec_5b[ 0]);
    ASSERT_EQUAL(vec_6b.x,  vec_6b[ 0]);
    ASSERT_EQUAL(vec_7b.x,  vec_7b[ 0]);
    ASSERT_EQUAL(vec_8b.x,  vec_8b[ 0]);
    ASSERT_EQUAL(vec_16b.x, vec_16b[0]);

    ASSERT_EQUAL(vec_1b.y,  vec_1b[ 1]);
    ASSERT_EQUAL(vec_2b.y,  vec_2b[ 1]);
    ASSERT_EQUAL(vec_3b.y,  vec_3b[ 1]);
    ASSERT_EQUAL(vec_4b.y,  vec_4b[ 1]);
    ASSERT_EQUAL(vec_5b.y,  vec_5b[ 1]);
    ASSERT_EQUAL(vec_6b.y,  vec_6b[ 1]);
    ASSERT_EQUAL(vec_7b.y,  vec_7b[ 1]);
    ASSERT_EQUAL(vec_8b.y,  vec_8b[ 1]);
    ASSERT_EQUAL(vec_16b.y, vec_16b[1]);

    ASSERT_EQUAL(vec_1b.z,  vec_1b[ 2]);
    ASSERT_EQUAL(vec_2b.z,  vec_2b[ 2]);
    ASSERT_EQUAL(vec_3b.z,  vec_3b[ 2]);
    ASSERT_EQUAL(vec_4b.z,  vec_4b[ 2]);
    ASSERT_EQUAL(vec_5b.z,  vec_5b[ 2]);
    ASSERT_EQUAL(vec_6b.z,  vec_6b[ 2]);
    ASSERT_EQUAL(vec_7b.z,  vec_7b[ 2]);
    ASSERT_EQUAL(vec_8b.z,  vec_8b[ 2]);
    ASSERT_EQUAL(vec_16b.z, vec_16b[2]);

    ASSERT_EQUAL(vec_1b.w,  vec_1b[ 3]);
    ASSERT_EQUAL(vec_2b.w,  vec_2b[ 3]);
    ASSERT_EQUAL(vec_3b.w,  vec_3b[ 3]);
    ASSERT_EQUAL(vec_4b.w,  vec_4b[ 3]);
    ASSERT_EQUAL(vec_5b.w,  vec_5b[ 3]);
    ASSERT_EQUAL(vec_6b.w,  vec_6b[ 3]);
    ASSERT_EQUAL(vec_7b.w,  vec_7b[ 3]);
    ASSERT_EQUAL(vec_8b.w,  vec_8b[ 3]);
    ASSERT_EQUAL(vec_16b.w, vec_16b[3]);

    ASSERT_EQUAL(vec_1b.x,  *(vec_1b.data()  + 0));
    ASSERT_EQUAL(vec_2b.x,  *(vec_2b.data()  + 0));
    ASSERT_EQUAL(vec_3b.x,  *(vec_3b.data()  + 0));
    ASSERT_EQUAL(vec_4b.x,  *(vec_4b.data()  + 0));
    ASSERT_EQUAL(vec_5b.x,  *(vec_5b.data()  + 0));
    ASSERT_EQUAL(vec_6b.x,  *(vec_6b.data()  + 0));
    ASSERT_EQUAL(vec_7b.x,  *(vec_7b.data()  + 0));
    ASSERT_EQUAL(vec_8b.x,  *(vec_8b.data()  + 0));
    ASSERT_EQUAL(vec_16b.x, *(vec_16b.data() + 0));

    ASSERT_EQUAL(vec_1b.y,  *(vec_1b.data()  + 1));
    ASSERT_EQUAL(vec_2b.y,  *(vec_2b.data()  + 1));
    ASSERT_EQUAL(vec_3b.y,  *(vec_3b.data()  + 1));
    ASSERT_EQUAL(vec_4b.y,  *(vec_4b.data()  + 1));
    ASSERT_EQUAL(vec_5b.y,  *(vec_5b.data()  + 1));
    ASSERT_EQUAL(vec_6b.y,  *(vec_6b.data()  + 1));
    ASSERT_EQUAL(vec_7b.y,  *(vec_7b.data()  + 1));
    ASSERT_EQUAL(vec_8b.y,  *(vec_8b.data()  + 1));
    ASSERT_EQUAL(vec_16b.y, *(vec_16b.data() + 1));

    ASSERT_EQUAL(vec_1b.z,  *(vec_1b.data()  + 2));
    ASSERT_EQUAL(vec_2b.z,  *(vec_2b.data()  + 2));
    ASSERT_EQUAL(vec_3b.z,  *(vec_3b.data()  + 2));
    ASSERT_EQUAL(vec_4b.z,  *(vec_4b.data()  + 2));
    ASSERT_EQUAL(vec_5b.z,  *(vec_5b.data()  + 2));
    ASSERT_EQUAL(vec_6b.z,  *(vec_6b.data()  + 2));
    ASSERT_EQUAL(vec_7b.z,  *(vec_7b.data()  + 2));
    ASSERT_EQUAL(vec_8b.z,  *(vec_8b.data()  + 2));
    ASSERT_EQUAL(vec_16b.z, *(vec_16b.data() + 2));

    ASSERT_EQUAL(vec_1b.w,  *(vec_1b.data()  + 3));
    ASSERT_EQUAL(vec_2b.w,  *(vec_2b.data()  + 3));
    ASSERT_EQUAL(vec_3b.w,  *(vec_3b.data()  + 3));
    ASSERT_EQUAL(vec_4b.w,  *(vec_4b.data()  + 3));
    ASSERT_EQUAL(vec_5b.w,  *(vec_5b.data()  + 3));
    ASSERT_EQUAL(vec_6b.w,  *(vec_6b.data()  + 3));
    ASSERT_EQUAL(vec_7b.w,  *(vec_7b.data()  + 3));
    ASSERT_EQUAL(vec_8b.w,  *(vec_8b.data()  + 3));
    ASSERT_EQUAL(vec_16b.w, *(vec_16b.data() + 3));
}

GRACE_HOST_DEVICE void test_vector4_accessors(
    const Vector4Ptrs vec_ptrs,
    const size_t index)
{
    test_vector4_accessors_impl(
        *(vec_ptrs.vec_1b  + index),
        *(vec_ptrs.vec_2b  + index),
        *(vec_ptrs.vec_3b  + index),
        *(vec_ptrs.vec_4b  + index),
        *(vec_ptrs.vec_5b  + index),
        *(vec_ptrs.vec_6b  + index),
        *(vec_ptrs.vec_7b  + index),
        *(vec_ptrs.vec_8b  + index),
        *(vec_ptrs.vec_16b + index)
    );
}

GRACE_HOST_DEVICE void compare_vector4s_impl(
    const vec4_1T& vecA_1b,   const vec4_1T& vecB_1b,
    const vec4_2T& vecA_2b,   const vec4_2T& vecB_2b,
    const vec4_3T& vecA_3b,   const vec4_3T& vecB_3b,
    const vec4_4T& vecA_4b,   const vec4_4T& vecB_4b,
    const vec4_5T& vecA_5b,   const vec4_5T& vecB_5b,
    const vec4_6T& vecA_6b,   const vec4_6T& vecB_6b,
    const vec4_7T& vecA_7b,   const vec4_7T& vecB_7b,
    const vec4_8T& vecA_8b,   const vec4_8T& vecB_8b,
    const vec4_16T& vecA_16b, const vec4_16T vecB_16b)
{
    ASSERT_EQUAL(vecA_1b.x,  vecB_1b.x);
    ASSERT_EQUAL(vecA_2b.x,  vecB_2b.x);
    ASSERT_EQUAL(vecA_3b.x,  vecB_3b.x);
    ASSERT_EQUAL(vecA_4b.x,  vecB_4b.x);
    ASSERT_EQUAL(vecA_5b.x,  vecB_5b.x);
    ASSERT_EQUAL(vecA_6b.x,  vecB_6b.x);
    ASSERT_EQUAL(vecA_7b.x,  vecB_7b.x);
    ASSERT_EQUAL(vecA_8b.x,  vecB_8b.x);
    ASSERT_EQUAL(vecA_16b.x, vecB_16b.x);

    ASSERT_EQUAL(vecA_1b.y,  vecB_1b.y);
    ASSERT_EQUAL(vecA_2b.y,  vecB_2b.y);
    ASSERT_EQUAL(vecA_3b.y,  vecB_3b.y);
    ASSERT_EQUAL(vecA_4b.y,  vecB_4b.y);
    ASSERT_EQUAL(vecA_5b.y,  vecB_5b.y);
    ASSERT_EQUAL(vecA_6b.y,  vecB_6b.y);
    ASSERT_EQUAL(vecA_7b.y,  vecB_7b.y);
    ASSERT_EQUAL(vecA_8b.y,  vecB_8b.y);
    ASSERT_EQUAL(vecA_16b.y, vecB_16b.y);

    ASSERT_EQUAL(vecA_1b.z,  vecB_1b.z);
    ASSERT_EQUAL(vecA_2b.z,  vecB_2b.z);
    ASSERT_EQUAL(vecA_3b.z,  vecB_3b.z);
    ASSERT_EQUAL(vecA_4b.z,  vecB_4b.z);
    ASSERT_EQUAL(vecA_5b.z,  vecB_5b.z);
    ASSERT_EQUAL(vecA_6b.z,  vecB_6b.z);
    ASSERT_EQUAL(vecA_7b.z,  vecB_7b.z);
    ASSERT_EQUAL(vecA_8b.z,  vecB_8b.z);
    ASSERT_EQUAL(vecA_16b.z, vecB_16b.z);

    ASSERT_EQUAL(vecA_1b.w,  vecB_1b.w);
    ASSERT_EQUAL(vecA_2b.w,  vecB_2b.w);
    ASSERT_EQUAL(vecA_3b.w,  vecB_3b.w);
    ASSERT_EQUAL(vecA_4b.w,  vecB_4b.w);
    ASSERT_EQUAL(vecA_5b.w,  vecB_5b.w);
    ASSERT_EQUAL(vecA_6b.w,  vecB_6b.w);
    ASSERT_EQUAL(vecA_7b.w,  vecB_7b.w);
    ASSERT_EQUAL(vecA_8b.w,  vecB_8b.w);
    ASSERT_EQUAL(vecA_16b.w, vecB_16b.w);
}

GRACE_HOST_DEVICE void compare_vector4s(
    const Vector4Ptrs vecA_ptrs,
    const Vector4Ptrs vecB_ptrs,
    const size_t index)
{
    compare_vector4s_impl(
        *(vecA_ptrs.vec_1b  + index), *(vecB_ptrs.vec_1b  + index),
        *(vecA_ptrs.vec_2b  + index), *(vecB_ptrs.vec_2b  + index),
        *(vecA_ptrs.vec_3b  + index), *(vecB_ptrs.vec_3b  + index),
        *(vecA_ptrs.vec_4b  + index), *(vecB_ptrs.vec_4b  + index),
        *(vecA_ptrs.vec_5b  + index), *(vecB_ptrs.vec_5b  + index),
        *(vecA_ptrs.vec_6b  + index), *(vecB_ptrs.vec_6b  + index),
        *(vecA_ptrs.vec_7b  + index), *(vecB_ptrs.vec_7b  + index),
        *(vecA_ptrs.vec_8b  + index), *(vecB_ptrs.vec_8b  + index),
        *(vecA_ptrs.vec_16b + index), *(vecB_ptrs.vec_16b + index)
    );
}
