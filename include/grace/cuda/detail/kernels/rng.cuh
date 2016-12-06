#pragma once

#include <curand_kernel.h>

template <typename PRNGStateT>
__global__ void init_PRNG_states_kernel(
    PRNGStateT* const prng_states,
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
