#pragma once

// This should be first so assert() is correctly defined.
#include "helper-unit/assert_macros.h"
#include "helper-unit/test_types.h"

#include "grace/config.h"
#include "grace/vector.h"


#include <cstdlib>

#define BLOAT_FACTOR_NONE 1.0
#define BLOAT_FACTOR_SMALL 1.2
#define BLOAT_FACTOR_LARGE 1.5
#define NUM_TEST_VECTORS 9

typedef grace::Vector<3, Byte1T> vec3_1T;
typedef grace::Vector<3, Byte2T> vec3_2T;
typedef grace::Vector<3, Byte3T> vec3_3T;
typedef grace::Vector<3, Byte4T> vec3_4T;
typedef grace::Vector<3, Byte5T> vec3_5T;
typedef grace::Vector<3, Byte6T> vec3_6T;
typedef grace::Vector<3, Byte7T> vec3_7T;
typedef grace::Vector<3, Byte8T> vec3_8T;
typedef grace::Vector<3, Byte16T> vec3_16T;

struct Vector3Ptrs
{
    grace::Vector<3, Byte1T>* vec_1b;
    grace::Vector<3, Byte2T>* vec_2b;
    grace::Vector<3, Byte3T>* vec_3b;
    grace::Vector<3, Byte4T>* vec_4b;
    grace::Vector<3, Byte5T>* vec_5b;
    grace::Vector<3, Byte6T>* vec_6b;
    grace::Vector<3, Byte7T>* vec_7b;
    grace::Vector<3, Byte8T>* vec_8b;
    grace::Vector<3, Byte16T>* vec_16b;
    size_t n;
};

template <typename T>
GRACE_HOST_DEVICE double bloat(const grace::Vector<3, T>& vec)
{
    return sizeof(vec) / static_cast<double>((3 * sizeof(T)));
}

template <typename T>
GRACE_HOST_DEVICE int next_lowest_multiple(const grace::Vector<3, T>& vec)
{
    return (int)(GRACE_ALIGNOF(vec)) * ((int)(sizeof(vec) / GRACE_ALIGNOF(vec)) - 1);
}

GRACE_HOST_DEVICE void set_vector3_data(const Vector3Ptrs vec_ptrs,
                                        const int index, const size_t n_total)
{
    int init = 3 * n_total * index;

    *(vec_ptrs.vec_1b  + index) = vec3_1T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_2b  + index) = vec3_2T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_3b  + index) = vec3_3T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_4b  + index) = vec3_4T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_5b  + index) = vec3_5T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_6b  + index) = vec3_6T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_7b  + index) = vec3_7T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_8b  + index) = vec3_8T( init+1, init+2, init+3);
    init += 3;
    *(vec_ptrs.vec_16b + index) = vec3_16T(init+1, init+2, init+3);
}

GRACE_HOST_DEVICE void test_vector3_size_impl(
    const vec3_1T  vec_1b,
    const vec3_2T  vec_2b,
    const vec3_3T  vec_3b,
    const vec3_4T  vec_4b,
    const vec3_5T  vec_5b,
    const vec3_6T  vec_6b,
    const vec3_7T  vec_7b,
    const vec3_8T  vec_8b,
    const vec3_16T vec_16b)
{
    ASSERT_LESS_THAN_EQUAL(bloat(vec_16b), BLOAT_FACTOR_NONE);

    ASSERT_LESS_THAN_EQUAL(bloat(vec_5b),  BLOAT_FACTOR_SMALL);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_6b),  BLOAT_FACTOR_SMALL);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_7b),  BLOAT_FACTOR_SMALL);

    ASSERT_LESS_THAN_EQUAL(bloat(vec_1b),  BLOAT_FACTOR_LARGE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_2b),  BLOAT_FACTOR_LARGE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_3b),  BLOAT_FACTOR_LARGE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_4b),  BLOAT_FACTOR_LARGE);
    ASSERT_LESS_THAN_EQUAL(bloat(vec_8b),  BLOAT_FACTOR_LARGE);
}

GRACE_HOST_DEVICE void test_vector3_size()
{
    // Create on stack.
    test_vector3_size_impl(
        vec3_1T(),
        vec3_2T(),
        vec3_3T(),
        vec3_4T(),
        vec3_5T(),
        vec3_6T(),
        vec3_7T(),
        vec3_8T(),
        vec3_16T()
    );
}

GRACE_HOST_DEVICE void test_vector3_padding_impl(
    const vec3_1T  vec_1b,
    const vec3_2T  vec_2b,
    const vec3_3T  vec_3b,
    const vec3_4T  vec_4b,
    const vec3_5T  vec_5b,
    const vec3_6T  vec_6b,
    const vec3_7T  vec_7b,
    const vec3_8T  vec_8b,
    const vec3_16T vec_16b)
{
    // sizeof(vec) should be the smallest-possible multiple of alignof(vec).

    ASSERT_ZERO(sizeof(vec_1b) %  GRACE_ALIGNOF(vec_1b ));
    ASSERT_ZERO(sizeof(vec_2b) %  GRACE_ALIGNOF(vec_2b ));
    ASSERT_ZERO(sizeof(vec_3b) %  GRACE_ALIGNOF(vec_3b ));
    ASSERT_ZERO(sizeof(vec_4b) %  GRACE_ALIGNOF(vec_4b ));
    ASSERT_ZERO(sizeof(vec_5b) %  GRACE_ALIGNOF(vec_5b ));
    ASSERT_ZERO(sizeof(vec_6b) %  GRACE_ALIGNOF(vec_6b ));
    ASSERT_ZERO(sizeof(vec_7b) %  GRACE_ALIGNOF(vec_7b ));
    ASSERT_ZERO(sizeof(vec_8b) %  GRACE_ALIGNOF(vec_8b ));
    ASSERT_ZERO(sizeof(vec_16b) % GRACE_ALIGNOF(vec_16b));

    ASSERT_LESS_THAN(next_lowest_multiple(vec_1b), (int)(3 * sizeof(Byte1T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_2b), (int)(3 * sizeof(Byte2T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_3b), (int)(3 * sizeof(Byte3T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_4b), (int)(3 * sizeof(Byte4T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_5b), (int)(3 * sizeof(Byte5T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_6b), (int)(3 * sizeof(Byte6T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_7b), (int)(3 * sizeof(Byte7T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_8b), (int)(3 * sizeof(Byte8T)));
    ASSERT_LESS_THAN(next_lowest_multiple(vec_16b),(int)(3 * sizeof(Byte16T)));
}

GRACE_HOST_DEVICE void test_vector3_padding()
{
    // Create on stack.
    test_vector3_padding_impl(
        vec3_1T(),
        vec3_2T(),
        vec3_3T(),
        vec3_4T(),
        vec3_5T(),
        vec3_6T(),
        vec3_7T(),
        vec3_8T(),
        vec3_16T()
    );
}

GRACE_HOST_DEVICE void test_vector3_accessors_impl(
    const vec3_1T& vec_1b,
    const vec3_2T& vec_2b,
    const vec3_3T& vec_3b,
    const vec3_4T& vec_4b,
    const vec3_5T& vec_5b,
    const vec3_6T& vec_6b,
    const vec3_7T& vec_7b,
    const vec3_8T& vec_8b,
    const vec3_16T& vec_16b)
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
}

GRACE_HOST_DEVICE void test_vector3_accessors(
    const Vector3Ptrs vec_ptrs,
    const size_t index)
{
    test_vector3_accessors_impl(
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

GRACE_HOST_DEVICE void compare_vector3s_impl(
    const vec3_1T& vecA_1b,   const vec3_1T& vecB_1b,
    const vec3_2T& vecA_2b,   const vec3_2T& vecB_2b,
    const vec3_3T& vecA_3b,   const vec3_3T& vecB_3b,
    const vec3_4T& vecA_4b,   const vec3_4T& vecB_4b,
    const vec3_5T& vecA_5b,   const vec3_5T& vecB_5b,
    const vec3_6T& vecA_6b,   const vec3_6T& vecB_6b,
    const vec3_7T& vecA_7b,   const vec3_7T& vecB_7b,
    const vec3_8T& vecA_8b,   const vec3_8T& vecB_8b,
    const vec3_16T& vecA_16b, const vec3_16T vecB_16b)
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
}

GRACE_HOST_DEVICE void compare_vector3s(
    const Vector3Ptrs vecA_ptrs,
    const Vector3Ptrs vecB_ptrs,
    const size_t index)
{
    compare_vector3s_impl(
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
