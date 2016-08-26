#include "helper-unit/assert_macros.h"

#include "grace/aligned_malloc.h"

int main(void)
{
    float* ptr_4
        = static_cast<float*>(grace::aligned_malloc(100 * sizeof(float), 4));

    float* ptr_8
        = static_cast<float*>(grace::aligned_malloc(100 * sizeof(float), 8));

    float* ptr_16
        = static_cast<float*>(grace::aligned_malloc(100 * sizeof(float), 16));

    float* ptr_32
        = static_cast<float*>(grace::aligned_malloc(100 * sizeof(float), 32));

    float* ptr_64
        = static_cast<float*>(grace::aligned_malloc(100 * sizeof(float), 64));

    float* ptr_128
        = static_cast<float*>(grace::aligned_malloc(100 * sizeof(float), 128));

    float* ptr_256
        = static_cast<float*>(grace::aligned_malloc(100 * sizeof(float), 256));


    ASSERT_NOT_EQUAL_PTR((void*)ptr_4,   (void*)NULL);
    ASSERT_NOT_EQUAL_PTR((void*)ptr_8,   (void*)NULL);
    ASSERT_NOT_EQUAL_PTR((void*)ptr_16,  (void*)NULL);
    ASSERT_NOT_EQUAL_PTR((void*)ptr_32,  (void*)NULL);
    ASSERT_NOT_EQUAL_PTR((void*)ptr_64,  (void*)NULL);
    ASSERT_NOT_EQUAL_PTR((void*)ptr_128, (void*)NULL);
    ASSERT_NOT_EQUAL_PTR((void*)ptr_256, (void*)NULL);

    ASSERT_ZERO((unsigned long long)ptr_4   % 4);
    ASSERT_ZERO((unsigned long long)ptr_8   % 8);
    ASSERT_ZERO((unsigned long long)ptr_16  % 16);
    ASSERT_ZERO((unsigned long long)ptr_32  % 32);
    ASSERT_ZERO((unsigned long long)ptr_64  % 64);
    ASSERT_ZERO((unsigned long long)ptr_128 % 128);
    ASSERT_ZERO((unsigned long long)ptr_256 % 256);

    grace::aligned_free(ptr_4);
    grace::aligned_free(ptr_8);
    grace::aligned_free(ptr_16);
    grace::aligned_free(ptr_32);
    grace::aligned_free(ptr_64);
    grace::aligned_free(ptr_128);
    grace::aligned_free(ptr_256);

    return 0;
}
