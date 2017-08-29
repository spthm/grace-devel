#pragma once

// No grace/cuda/error.cuh include.
// This should only ever be included by grace/cuda/error.cuh.

#include "grace/config.h"

#include "cuda_runtime_api.h"

#include <sstream>
#include <string>

namespace grace {

namespace detail {

GRACE_HOST void throw_on_cudart_error(
    cudaError_t code,
    const char* fname,
    int line)
{
    if (code != cudaSuccess) {
        std::stringstream fn_line_ss;
        fn_line_ss << fname << "(" << line << ")";
        std::string fn_line;
        fn_line_ss >> fn_line;
        throw cuda_runtime_error(code, fn_line);
    }
}

GRACE_HOST void cuda_kernel_check(const char* fname, int line)
{
    // Check launch was successful.
    throw_on_cudart_error(cudaPeekAtLastError(), fname, line);

    // Check kernel successfully returned, if in DEBUG mode. Otherwise, don't
    // force synchronization --- any errors will be caught at some point in
    // the future, after the kernel returns.
    #ifdef GRACE_DEBUG
    throw_on_cudart_error(cudaDeviceSynchronize(), fname, line);
    #endif
}

} // namespace detail

} // namespace grace

// Wrap around all calls to CUDA functions to handle errors.
#define GRACE_CUDA_CHECK(code) { grace::detail::throw_on_cudart_error((code), __FILE__, __LINE__); }
#define GRACE_CUDA_KERNEL_CHECK() { grace::detail::cuda_kernel_check(__FILE__, __LINE__); }
