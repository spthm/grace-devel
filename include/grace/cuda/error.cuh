#pragma once

#include "grace/config.h"

#include "cuda_runtime_api.h"

#include <stdexcept>
#include <string>

namespace grace {

class cuda_runtime_error : public std::runtime_error
{
public:
    // TODO: C++11 std::runtime also has a const char* argument constructor.
    GRACE_HOST
    explicit cuda_runtime_error(cudaError_t code)
        : std::runtime_error(cudaGetErrorString(code)) {}

    // what_arg should be "file_name(line)" or similar; the cuda API error
    // description for code is automatically appended.
    GRACE_HOST
    cuda_runtime_error(cudaError_t code,
                       const std::string& what_arg)
        : std::runtime_error(what_arg + ": " + cudaGetErrorString(code)) {}
};

} // namespace grace

#include "grace/cuda/detail/error-inl.cuh"
