// This should be first so assert() is correctly defined.
#include "helper-unit/assert_macros.h"

#include "grace/cuda/error.cuh"

__global__ void illegal_memory_access_kernel(int *nullptr_a, int *nullptr_b)
{
  *nullptr_a = *nullptr_b;
}

__global__ void invalid_configuration_kernel() {}


int main(int argc, char **argv)
{
    bool thrown_exception;


    thrown_exception = false;
    try
    {
        illegal_memory_access_kernel<<<1,1>>>((int*)0, (int*)0);
        // Need to synchronize first because this is a post-launch failure.
        cudaDeviceSynchronize();
        GRACE_CUDA_KERNEL_CHECK();
    }
    catch (grace::cuda_runtime_error)
    {
        thrown_exception = true;
        cudaDeviceReset();
    }
    ASSERT_EQUAL(thrown_exception, true);


    thrown_exception = false;
    try
    {
        invalid_configuration_kernel<<<1,(1<<30)>>>();
        // Do not need to synchronize first; this is a launch-time failure.
        GRACE_CUDA_KERNEL_CHECK();
    }
    catch (grace::cuda_runtime_error)
    {
        thrown_exception = true;
        cudaDeviceReset();
    }
    ASSERT_EQUAL(thrown_exception, true);


    thrown_exception = false;
    try
    {
        GRACE_CUDA_CHECK(cudaMemcpy((void*)0, (void*)0, 1,
                                    cudaMemcpyDeviceToDevice));
    }
    catch (grace::cuda_runtime_error)
    {
        thrown_exception = true;
        cudaDeviceReset();
    }
    ASSERT_EQUAL(thrown_exception, true);
}
