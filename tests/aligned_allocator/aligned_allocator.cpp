// This should be first so assert() is correctly defined.
#include "helper-unit/assert_macros.h"

#include "grace/aligned_allocator.h"

#include <vector>

int main(void)
{
    std::vector<float, grace::aligned_allocator<float, 4> >   vec_4(100);
    std::vector<float, grace::aligned_allocator<float, 8> >   vec_8(100);
    std::vector<float, grace::aligned_allocator<float, 16> >  vec_16(100);
    std::vector<float, grace::aligned_allocator<float, 32> >  vec_32(100);
    std::vector<float, grace::aligned_allocator<float, 64> >  vec_64(100);
    std::vector<float, grace::aligned_allocator<float, 128> > vec_128(100);
    std::vector<float, grace::aligned_allocator<float, 256> > vec_256(100);

    float* ptr_4   = &vec_4.front();
    float* ptr_8   = &vec_8.front();
    float* ptr_16  = &vec_16.front();
    float* ptr_32  = &vec_32.front();
    float* ptr_64  = &vec_64.front();
    float* ptr_128 = &vec_128.front();
    float* ptr_256 = &vec_256.front();

    ASSERT_ZERO((unsigned long long)ptr_4   % 4);
    ASSERT_ZERO((unsigned long long)ptr_8   % 8);
    ASSERT_ZERO((unsigned long long)ptr_16  % 16);
    ASSERT_ZERO((unsigned long long)ptr_32  % 32);
    ASSERT_ZERO((unsigned long long)ptr_64  % 64);
    ASSERT_ZERO((unsigned long long)ptr_128 % 128);
    ASSERT_ZERO((unsigned long long)ptr_256 % 256);

    return 0;
}
