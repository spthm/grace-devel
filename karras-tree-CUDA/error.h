#pragma once

#include "types.h"

#include <cstdio>

// Wrap around all calls to CUDA functions to handle errors.
#define GRACE_CUDA_CHECK(code) { grace::cuda_error_check((code), __FILE__, __LINE__); }
#define GRACE_KERNEL_CHECK() { grace::cuda_kernel_check(__FILE__, __LINE__); }

namespace grace {

GRACE_HOST void cuda_error_check(
    cudaError_t code,
    const char* file,
    int line,
    bool terminate=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error:\nMsg:  %s\nFile: %s @ line %d\n",
                cudaGetErrorString(code), file, line);

        if (terminate)
            exit(code);
    }
}

GRACE_HOST void cuda_kernel_check(const char* file, int line, bool terminate=true)
{
    cuda_error_check(cudaPeekAtLastError(), file, line, terminate);

    #ifdef GRACE_DEBUG
    cuda_error_check(cudaDeviceSynchronize(), file, line, terminate);
    #endif
}

} // namespace grace
