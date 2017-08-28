#pragma once

#include "grace/config.h"

#include <assert.h>
#include <iostream>

#ifdef GRACE_DEBUG
// assert(a > b && "Helpful message") generates warnings from nvcc. The below
// is a more portable, slightly less useful alternative.
#define GRACE_ASSERT_NOMSG(predicate) { assert(predicate); }
#define GRACE_ASSERT_MSG(predicate, err) { const bool err = true; assert(err && (predicate)); }
#define GRACE_SELECT_ASSERT(arg1, arg2, ASSERT_MACRO, ...) ASSERT_MACRO

// Usage:
// GRACE_ASSERT(a > b);
// GRACE_ASSERT(a > b, some_unused_variable_name_as_error_message);
// Where the error 'message' must be a valid, unused variable name.
#define GRACE_MSVC_VAARGS_FIX( x ) x
#define GRACE_ASSERT(...) GRACE_MSVC_VAARGS_FIX(GRACE_SELECT_ASSERT(__VA_ARGS__, GRACE_ASSERT_MSG, GRACE_ASSERT_NOMSG)(__VA_ARGS__))
#else
#define GRACE_ASSERT(...)
#endif // GRACE_DEBUG

template <bool> struct grace_static_assert;
template <> struct grace_static_assert<true> {};
#if defined(static_assert) || __STDC_VERSION >= 201112L
#define GRACE_STATIC_ASSERT(predicate, msg) { static_assert(predicate, msg); }
#else
#define GRACE_STATIC_ASSERT(predicate, ignore) { grace_static_assert<predicate>(); }
#endif

#define GRACE_GOT_TO() std::cerr << "At " << __FILE__ << "@" << __LINE__ << std::endl;


#ifdef __CUDACC__

#include "cuda_runtime_api.h"

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
        std::cerr << "**** GRACE CUDA Error ****" << std::endl
                  << "File:  " << file << std::endl
                  << "Line:  " << line << std::endl
                  << "Error: " << cudaGetErrorString(code) << std::endl;

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

#endif // ifdef __CUDACC__
