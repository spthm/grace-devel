#include "print_table.h"

#include "grace/cuda/detail/prngstates-inl.cuh"

#include "helper/cuda_timer.cuh"

#include <thrust/device_vector.h>

#define BITMASK 0xf

template <unsigned int bitmask, typename RngDeviceStatesT>
__global__ void generate_randoms_kernel(
    RngDeviceStatesT states,
    const size_t n,
    unsigned int* const results)
{
    typename RngDeviceStatesT::state_type state = states.load_state();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int count = 0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        unsigned int x = curand(&state);
        if (bitmask) {
          if ((x & bitmask) == bitmask) { ++count; }
        }
    }

    states.save_state(state);
    if (bitmask) {
      results[tid] = count;
    }
}

template <unsigned int bitmask, typename RngStatesT>
void generate_randoms(
    RngStatesT& states,
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
    generate_randoms_kernel<bitmask>
    <<<NT, num_blocks>>>(
        states.device_states(),
        n,
        thrust::raw_pointer_cast(d_results.data()));
}

typedef grace::detail::RngStates<curandStatePhilox4_32_10_t> PhiloxStates; // GRACE default
typedef grace::detail::RngStates<curandStateXORWOW_t> XORWOWStates; // CUDA default
typedef grace::detail::RngStates<curandStateMRG32k3a_t> MRG32States;

int main(int argc, char* argv[])
{
    cout_init();

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
    double fraction_set[num_generators];
    for (size_t n = 1000; n < max_n + 1; n *= 10)
    {
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

            if (i >= 0) {
                generate_randoms<0>(philox_states, n, d_results);
                rand_timings[PHILOX] += timer.split();
            }
            else {
                generate_randoms<BITMASK>(philox_states, n, d_results);
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                fraction_set[PHILOX] = (double)tot / n;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }

            if (i >= 0) {
                generate_randoms<0>(xorwow_states, n, d_results);
                rand_timings[XORWOW] += timer.split();
            }
            else {
                generate_randoms<BITMASK>(xorwow_states, n, d_results);
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                fraction_set[XORWOW] = (double)tot / n;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }

            if (i >= 0) {
                generate_randoms<0>(mrg32_states, n, d_results);
                rand_timings[MRG32] += timer.split();
            }
            else {
                generate_randoms<BITMASK>(mrg32_states, n, d_results);
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                fraction_set[MRG32] = (double)tot / n;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }
        }

        print_header(n);
        print_row(PHILOX, fraction_set[PHILOX], philox_states.size_bytes(),
                  init_timings[PHILOX], rand_timings[PHILOX]);
        print_row(XORWOW, fraction_set[XORWOW], xorwow_states.size_bytes(),
                  init_timings[XORWOW], rand_timings[XORWOW]);
        print_row(MRG32, fraction_set[MRG32], mrg32_states.size_bytes(),
                  init_timings[MRG32], rand_timings[MRG32]);
        print_footer();

        // std::cout << "  States' size: " << std::setw(9)
        //           << philox_states.size_bytes() / 1024. / 1024.
        //           << " MiB (PHILOX)" << std::endl
        //           << "  States' size: " << std::setw(9)
        //           << xorwow_states.size_bytes() / 1024. / 1024.
        //           << " MiB (XORWOW)" << std::endl
        //           << "  States' size: " << std::setw(9)
        //           << mrg32_states.size_bytes() / 1024. / 1024.
        //           << " MiB (MRG32)" << std::endl;
        // std::cout << "  Time to init: " << std::setw(10)
        //           << init_timings[PHILOX] / n_iter << " ms (PHILOX)"
        //           << std::endl
        //           << "  Time to init: " << std::setw(10)
        //           << init_timings[XORWOW] / n_iter << " ms (XORWOW)" <<
        //           std::endl
        //           << "  Time to init: " << std::setw(10)
        //           << init_timings[MRG32] / n_iter << " ms (MGR32)"
        //           << std::endl;
        // std::cout << "  Time to generate: " << std::setw(10)
        //           << rand_timings[PHILOX] / n_iter << " ms (PHILOX)"
        //           << std::endl
        //           << "  Time to generate: " << std::setw(10)
        //           << rand_timings[XORWOW] / n_iter << " ms (XORWOW)"
        //           << std::endl
        //           << "  Time to generate: " << std::setw(10)
        //           << rand_timings[MRG32] / n_iter << " ms (MGR32)"
        //           << std::endl
        //           << std::endl;
    }

    return EXIT_SUCCESS;
}
