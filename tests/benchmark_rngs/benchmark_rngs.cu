#include "grace/cuda/detail/PRNGStates-inl.cuh"

#include <thrust/device_vector.h>

#include <iostream>

template <typename RNGDeviceStatesT>
__global__ void generate_randoms_kernel(
    RNGDeviceStatesT states,
    const size_t n,
    unsigned int* const results)
{
    RNGDeviceStatesT::state_type state = states.load_state();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int count = 0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        unsigned int x = curand(&state);
        if (x & 0xf) { ++count; }
    }

    states.save_state(state);
    results[tid] = count;
}

template <typename RNGStatesT>
void generate_randoms(
    RNGStatesT& states,
    const size_t n,
    thrust::device_vector<unsigned int> d_results)
{
    const int NT = 128;
    cudaDeviceProp props;
    int device_id;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_id));
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    const int max_states = props.multiProcessorCount * props.maxThreadsPerMultiProcessor;

    // Round down! We cannot generate more threads than there are states.
    const int num_blocks = std::min(n, max_states) / NT;
    generate_randoms_kernel<<<NT, num_blocks>>>(
        states.device_states(),
        n,
        thrust::raw_pointer_cast(d_results.data()));
}

enum
{
    PHILOX,
    XORWOW,
    MRG32,
    num_generators
};

typedef grace::detail::RNGStates<curandStatePhilox4_32_10_t> PhiloxStates; // GRACE default
typedef grace::detail::RNGStates<curandStateXORWOW_t> XORWOWStates; // CUDA default
typedef grace::detail::RNGStates<curandStateMRG32k3a_t> MRG32States;

int main(int argc, char* argv[])
{
    size_t max_n = 100000000;
    int device_id = 0;

    if (argc > 1) {
        max_n = (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        device_id = (int)std::strtol(argv[2], NULL, 10);
    }
    cudaSetDevice(device_id);

    PhiloxStates philox_states;
    XORWOWStates xorwow_states;
    MRG32States mrg32_states;

    double init_timings[num_generators];
    double rand_timings[num_generators];
    for (size_t n = 1000; n < max_n + 1; n *= 10)
    {
        std::cout << "n: " << n << std::endl;

        for (size_t i = 0; i < num_generators; ++i)
        {
            init_timings[i] = 0.0;
            rand_timings[i] = 0.0;
        }

        for (size_t i = -1; i < n_iter; ++i)
        {
            // Note that the constructors call init_states for us, but we're
            // doing it here to collect timing info.
            CUDATimer timer;
            timer.start();

            philox_states.init_states();
            if (i >= 0) init_timings[PHILOX] += timer.split();

            xorwow_states.init_states();
            if (i >= 0) init_timings[XORWOWStates] += timer.split();

            mrg32_states.init_states();
            if (i >= 0) init_timings[MRG32] += timer.split();
        }

        thrust::device_vector<unsigned int> d_results(n);
        for (size_t i = -1; i < n_iter; ++i)
        {
            CUDATimer timer;
            timer.start();

            generate_randoms(philox_states, n, d_results);
            if (i >= 0) rand_timings[PHILOX] += timer.split();
            else
            {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                std::cout << "  Fraction of numbers with low four bits set: "
                          << (double)tot / n
                          << " (PHILOX)" << std::endl;
            }

            generate_randoms(xorwow_states, n, d_results);
            if (i >= 0) rand_timings[XORWOWStates] += timer.split();
            else
            {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                std::cout << "  Fraction of numbers with low four bits set: "
                          << (double)tot / n
                          << " (XORWOW)" << std::endl;
            }

            generate_randoms(mrg32_states, n, d_results);
            if (i >= 0) rand_timings[MRG32] += timer.split();
            else
            {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                std::cout << "  Fraction of numbers with low four bits set: "
                          << (double)tot / n
                          << " (MRG32)" << std::endl;
            }
        }

        std::cout << "  Time to init: " << init_timings[PHILOX] / n_iter
                  << " (PHILOX)" << std::endl
                  << "  Time to init: " << init_timings[XORWOW] / n_iter
                  << " (XORWOW)" << std::endl
                  << "  Time to init: " << init_timings[MRG32] / n_iter
                  << " (MGR32)" << std::endl;
        std::cout << "  Time to generate: " << rand_timings[PHILOX] / n_iter
                  << " (PHILOX)" << std::endl
                  << "  Time to generate: " << rand_timings[XORWOW] / n_iter
                  << " (XORWOW)" << std::endl
                  << "  Time to generate: " << rand_timings[MRG32] / n_iter
                  << " (MGR32)" << std::endl;
    }

    return EXIT_SUCCESS;
}
