#include "vector3_host.h"

#include "grace/types.h"
#include "grace/aligned_malloc.h"

// Forward declarations. These are not externally visible.

static void test_vector3_accessors_stack_host_impl();
static void test_vector3_accessors_heap_host_impl();


// The externally visible functions.

void test_vector3_size_host()
{
    test_vector3_size();
    printf("PASSED host Vector<3, T> size\n");
}

void test_vector3_padding_host()
{
    test_vector3_padding();
    printf("PASSED host Vector<3, T> padding\n");
}

void test_vector3_accessors_stack_host()
{
    test_vector3_accessors_stack_host_impl();
    printf("PASSED host stack-allocated Vector<3, T> accessors\n");
}

void test_vector3_accessors_heap_host()
{
    test_vector3_accessors_heap_host_impl();
    printf("PASSED host heap-allocated Vector<3, T> accessors\n");
}

Vector3Ptrs alloc_vector3s_host(const size_t n)
{
    Vector3Ptrs vec_ptrs;

    vec_ptrs.vec_1b  = (vec3_1T* )grace::aligned_malloc(n * sizeof(vec3_1T),
                                                        GRACE_ALIGNOF(vec3_1T));
    vec_ptrs.vec_2b  = (vec3_2T* )grace::aligned_malloc(n * sizeof(vec3_2T),
                                                        GRACE_ALIGNOF(vec3_2T));
    vec_ptrs.vec_3b  = (vec3_3T* )grace::aligned_malloc(n * sizeof(vec3_3T),
                                                        GRACE_ALIGNOF(vec3_3T));
    vec_ptrs.vec_4b  = (vec3_4T* )grace::aligned_malloc(n * sizeof(vec3_4T),
                                                        GRACE_ALIGNOF(vec3_4T));
    vec_ptrs.vec_5b  = (vec3_5T* )grace::aligned_malloc(n * sizeof(vec3_5T),
                                                        GRACE_ALIGNOF(vec3_5T));
    vec_ptrs.vec_6b  = (vec3_6T* )grace::aligned_malloc(n * sizeof(vec3_6T),
                                                        GRACE_ALIGNOF(vec3_6T));
    vec_ptrs.vec_7b  = (vec3_7T* )grace::aligned_malloc(n * sizeof(vec3_7T),
                                                        GRACE_ALIGNOF(vec3_7T));
    vec_ptrs.vec_8b  = (vec3_8T* )grace::aligned_malloc(n * sizeof(vec3_8T),
                                                        GRACE_ALIGNOF(vec3_8T));
    vec_ptrs.vec_16b = (vec3_16T*)grace::aligned_malloc(n * sizeof(vec3_16T),
                                                        GRACE_ALIGNOF(vec3_16T));

    vec_ptrs.n = n;

    return vec_ptrs;
}

void free_vectors_host(const Vector3Ptrs vec_ptrs)
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

void fill_vector3s_host(const Vector3Ptrs vec_ptrs)
{
    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        set_vector3_data(vec_ptrs, i, vec_ptrs.n);
    }
}

// Private functions.

static void test_vector3_accessors_stack_host_impl()
{
    Vector3Ptrs vec_ptrs;
    vec_ptrs.n = NUM_TEST_VECTORS;
    vec3_1T  vec_1b_array[ NUM_TEST_VECTORS];
    vec3_2T  vec_2b_array[ NUM_TEST_VECTORS];
    vec3_3T  vec_3b_array[ NUM_TEST_VECTORS];
    vec3_4T  vec_4b_array[ NUM_TEST_VECTORS];
    vec3_5T  vec_5b_array[ NUM_TEST_VECTORS];
    vec3_6T  vec_6b_array[ NUM_TEST_VECTORS];
    vec3_7T  vec_7b_array[ NUM_TEST_VECTORS];
    vec3_8T  vec_8b_array[ NUM_TEST_VECTORS];
    vec3_16T vec_16b_array[NUM_TEST_VECTORS];

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
        set_vector3_data(vec_ptrs, i, vec_ptrs.n);
    }


    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        test_vector3_accessors(vec_ptrs, i);
    }
}

static void test_vector3_accessors_heap_host_impl()
{
    Vector3Ptrs vec_ptrs = alloc_vector3s_host(NUM_TEST_VECTORS);

    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        set_vector3_data(vec_ptrs, i, vec_ptrs.n);
    }

    for (int i = 0; i < vec_ptrs.n; ++i)
    {
        test_vector3_accessors(vec_ptrs, i);
    }

    free_vectors_host(vec_ptrs);
}
