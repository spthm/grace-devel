#pragma once

// Due to a bug in the version of Thrust provided with CUDA 6, this must appear
// before #include <thrust/sort.h>
#include <curand_kernel.h>

__global__ void init_PRNG_kernel(
    curandState* const prng_states,
    const unsigned long long seed,
    const size_t N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Following the cuRAND documentation, each thread receives the same
        // seed value, no offset, and a *different* sequence value.
        // This should prevent any correlations if a single state is used to
        // generate multiple random numbers.
        curand_init(seed, tid, 0, &prng_states[tid]);
    }
}
