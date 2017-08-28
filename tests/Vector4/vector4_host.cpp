#include "vector4_host.h"

#include "grace/aligned_malloc.h"
#include "grace/config.h"

// Forward declarations. These are not externally visible.

static void test_vector4_accessors_stack_host_impl();
static void test_vector4_accessors_heap_host_impl();


// The externally visible functions.

void test_vector4_size_host()
{
    test_vector4_size();
    printf("PASSED host Vector<4, T> size\n");
}

void test_vector4_padding_host()
{
    test_vector4_padding();
    printf("PASSED host Vector<4, T> padding\n");
}

void test_vector4_accessors_stack_host()
{
    test_vector4_accessors_stack_host_impl();
    printf("PASSED host stack-allocated Vector<4, T> accessors\n");
}

void test_vector4_accessors_heap_host()
{
    test_vector4_accessors_heap_host_impl();
    printf("PASSED host heap-allocated Vector<4, T> accessors\n");
}

Vector4Ptrs alloc_vector4s_host(const size_t n)
{
    Vector4Ptrs vec_ptrs;

    vec_ptrs.vec_1b  = (vec4_1T* )grace::aligned_malloc(n * sizeof(vec4_1T),
                                                        GRACE_ALIGNOF(vec4_1T));
    vec_ptrs.vec_2b  = (vec4_2T* )grace::aligned_malloc(n * sizeof(vec4_2T),
                                                        GRACE_ALIGNOF(vec4_2T));
    vec_ptrs.vec_3b  = (vec4_3T* )grace::aligned_malloc(n * sizeof(vec4_3T),
                                                        GRACE_ALIGNOF(vec4_3T));
    vec_ptrs.vec_4b  = (vec4_4T* )grace::aligned_malloc(n * sizeof(vec4_4T),
                                                        GRACE_ALIGNOF(vec4_4T));
    vec_ptrs.vec_5b  = (vec4_5T* )grace::aligned_malloc(n * sizeof(vec4_5T),
                                                        GRACE_ALIGNOF(vec4_5T));
    vec_ptrs.vec_6b  = (vec4_6T* )grace::aligned_malloc(n * sizeof(vec4_6T),
                                                        GRACE_ALIGNOF(vec4_6T));
    vec_ptrs.vec_7b  = (vec4_7T* )grace::aligned_malloc(n * sizeof(vec4_7T),
                                                        GRACE_ALIGNOF(vec4_7T));
    vec_ptrs.vec_8b  = (vec4_8T* )grace::aligned_malloc(n * sizeof(vec4_8T),
                                                        GRACE_ALIGNOF(vec4_8T));
    vec_ptrs.vec_16b = (vec4_16T*)grace::aligned_malloc(n * sizeof(vec4_16T),
                                                        GRACE_ALIGNOF(vec4_16T));

    vec_ptrs.n = n;

    return vec_ptrs;
}

void free_vectors_host(const Vector4Ptrs vec_ptrs)
{
    free(vec_ptrs.vec_1b);
    free(vec_ptrs.vec_2b);
    free(vec_ptrs.vec_3b);
    free(vec_ptrs.vec_4b);
    free(vec_ptrs.vec_5b);
    free(vec_ptrs.vec_6b);
    free(vec_ptrs.vec_7b);
    free(vec_ptrs.vec_8b);
    free(vec_ptrs.vec_16b);
}

void fill_vector4s_host(const Vector4Ptrs vec_ptrs)
{
    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        set_vector4_data(vec_ptrs, i, vec_ptrs.n);
    }
}

// Private functions.

static void test_vector4_accessors_stack_host_impl()
{
    Vector4Ptrs vec_ptrs;
    vec_ptrs.n = NUM_TEST_VECTORS;
    vec4_1T  vec_1b_array[ NUM_TEST_VECTORS];
    vec4_2T  vec_2b_array[ NUM_TEST_VECTORS];
    vec4_3T  vec_3b_array[ NUM_TEST_VECTORS];
    vec4_4T  vec_4b_array[ NUM_TEST_VECTORS];
    vec4_5T  vec_5b_array[ NUM_TEST_VECTORS];
    vec4_6T  vec_6b_array[ NUM_TEST_VECTORS];
    vec4_7T  vec_7b_array[ NUM_TEST_VECTORS];
    vec4_8T  vec_8b_array[ NUM_TEST_VECTORS];
    vec4_16T vec_16b_array[NUM_TEST_VECTORS];

    vec_ptrs.vec_1b  = vec_1b_array;
    vec_ptrs.vec_2b  = vec_2b_array;
    vec_ptrs.vec_3b  = vec_3b_array;
    vec_ptrs.vec_4b  = vec_4b_array;
    vec_ptrs.vec_5b  = vec_5b_array;
    vec_ptrs.vec_6b  = vec_6b_array;
    vec_ptrs.vec_7b  = vec_7b_array;
    vec_ptrs.vec_8b  = vec_8b_array;
    vec_ptrs.vec_16b = vec_16b_array;

    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        set_vector4_data(vec_ptrs, i, vec_ptrs.n);
    }

    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        test_vector4_accessors(vec_ptrs, i);
    }
}

static void test_vector4_accessors_heap_host_impl()
{
    Vector4Ptrs vec_ptrs = alloc_vector4s_host(NUM_TEST_VECTORS);

    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        set_vector4_data(vec_ptrs, i, vec_ptrs.n);
    }

    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        test_vector4_accessors(vec_ptrs, i);
    }

    free_vectors_host(vec_ptrs);
}
