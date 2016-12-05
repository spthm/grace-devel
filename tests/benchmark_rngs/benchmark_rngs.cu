#include "grace/cuda/detail/PRNGStates-inl.cuh"

#include "helper/cuda_timer.cuh"

#include <thrust/device_vector.h>

#include <iomanip>
#include <iostream>

template <typename RNGDeviceStatesT>
__global__ void generate_randoms_kernel(
    RNGDeviceStatesT states,
    const size_t n,
    unsigned int* const results)
{
    typename RNGDeviceStatesT::state_type state = states.load_state();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int count = 0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        unsigned int x = curand(&state);
        if ((x & 0xf) == 0xf) { ++count; }
    }

    states.save_state(state);
    results[tid] = count;
}

template <typename RNGStatesT>
void generate_randoms(
    RNGStatesT& states,
    const size_t n,
    thrust::device_vector<unsigned int>& d_results)
{
    const int NT = 128;
    cudaDeviceProp props;
    int device_id;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_id));
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    const size_t max_states = props.multiProcessorCount
                                * props.maxThreadsPerMultiProcessor;

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
    std::cout.fill(' ');
    std::cout.setf(std::ios::fixed, std::ios::floatfield);

    size_t max_n = 100000000;
    int n_iter = 10;
    int device_id = 0;

    if (argc > 1) {
        max_n = (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        n_iter = (int)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        device_id = (int)std::strtol(argv[3], NULL, 10);
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

        for (int i = 0; i < num_generators; ++i)
        {
            init_timings[i] = 0.0;
            rand_timings[i] = 0.0;
        }

        for (int i = -1; i < n_iter; ++i)
        {
            // Note that the constructors call init_states for us, but we're
            // doing it here to collect timing info.
            CUDATimer timer;
            timer.start();

            philox_states.init_states();
            if (i >= 0) init_timings[PHILOX] += timer.split();

            xorwow_states.init_states();
            if (i >= 0) init_timings[XORWOW] += timer.split();

            mrg32_states.init_states();
            if (i >= 0) init_timings[MRG32] += timer.split();
        }

        thrust::device_vector<unsigned int> d_results(n);
        for (int i = -1; i < n_iter; ++i)
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
                          << std::setw(8) << (double)tot / n
                          << " (PHILOX)" << std::endl;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }

            generate_randoms(xorwow_states, n, d_results);
            if (i >= 0) rand_timings[XORWOW] += timer.split();
            else
            {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                std::cout << "  Fraction of numbers with low four bits set: "
                          << std::setw(8) << (double)tot / n
                          << " (XORWOW)" << std::endl;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }

            generate_randoms(mrg32_states, n, d_results);
            if (i >= 0) rand_timings[MRG32] += timer.split();
            else
            {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                std::cout << "  Fraction of numbers with low four bits set: "
                          << std::setw(8) << (double)tot / n
                          << " (MRG32)" << std::endl;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }
        }

        std::cout << "  Time to init: " << std::setw(10)
                  << init_timings[PHILOX] / n_iter << " ms (PHILOX)"
                  << std::endl
                  << "  Time to init: " << std::setw(10)
                  << init_timings[XORWOW] / n_iter << " ms (XORWOW)" <<
                  std::endl
                  << "  Time to init: " << std::setw(10)
                  << init_timings[MRG32] / n_iter << " ms (MGR32)"
                  << std::endl;
        std::cout << "  Time to generate: " << std::setw(10)
                  << rand_timings[PHILOX] / n_iter << " ms (PHILOX)"
                  << std::endl
                  << "  Time to generate: " << std::setw(10)
                  << rand_timings[XORWOW] / n_iter << " ms (XORWOW)"
                  << std::endl
                  << "  Time to generate: " << std::setw(10)
                  << rand_timings[MRG32] / n_iter << " ms (MGR32)"
                  << std::endl
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
