#include "print_table.h"

#include "grace/cuda/detail/prngstates-inl.cuh"

#include "helper/cuda_timer.cuh"

#include <thrust/device_vector.h>

// #define BITMASK 0x1
#define BITMASK 0x3
// #define BITMASK 0x7
// #define BITMASK 0xf

// XORWOW, MRG32
template <typename StateT>
__device__ float3 normal3(StateT& state)
{
    float2 xy = curand_normal2(&state);
    float z = curand_normal(&state);
    return make_float3(xy.x, xy.y, z);
}
// Philox is the only generator with a curand_normal4() function.
// This appears to be faster than the above at a blocksize of 128, but not at
// 256, on a Tesla M2090...
template <>
__device__ float3 normal3(curandStatePhilox4_32_10_t& state)
{
    float4 xyzw = curand_normal4(&state);
    return make_float3(xyzw.x, xyzw.y, xyzw.z);
}

template <typename RngDeviceStatesT>
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
        // Note that if nothing is done with the generated normal values, then
        // (at least in some cases), the compiler appears to optimize out the
        // conversion to normal, leaving only the state modification.
        // But we want to include any overhead specific to generating normal
        // values!
        // This is particularly important for Philox, where the state transition
        // is essentially an increment operation.
        float3 xyz = normal3(state);
        float t = fma(xyz.x,  xyz.y, xyz.z);
        unsigned int u = reinterpret_cast<unsigned int&>(t);
        if ((u & BITMASK) == BITMASK) { ++count; }
    }

    states.save_state(state);
    results[tid] = count;
}

template <typename RngStatesT>
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
    generate_randoms_kernel<<<NT, num_blocks>>>(
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

            // New state each time to (try to) avoid any clever compiler
            // optimization.
            philox_states.init_states(123456789 + i);
            if (i >= 0) init_timings[PHILOX] += timer.split();

            xorwow_states.init_states(123456789 + i);
            if (i >= 0) init_timings[XORWOW] += timer.split();

            mrg32_states.init_states(123456789 + i);
            if (i >= 0) init_timings[MRG32] += timer.split();
        }

        thrust::device_vector<unsigned int> d_results(n);
        for (int i = -1; i < n_iter; ++i)
        {
            CUDATimer timer;
            timer.start();

            generate_randoms(philox_states, n, d_results);
            if (i >= 0) {
              rand_timings[PHILOX] += timer.split();
            }
            else {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                fraction_set[PHILOX] = (double)tot / n;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }

            generate_randoms(xorwow_states, n, d_results);
            if (i >= 0) {
              rand_timings[XORWOW] += timer.split();
            }
            else {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                fraction_set[XORWOW] = (double)tot / n;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }

            generate_randoms(mrg32_states, n, d_results);
            if (i >= 0) {
              rand_timings[MRG32] += timer.split();
            }
            else {
                unsigned int tot = thrust::reduce(d_results.begin(),
                                                  d_results.end());
                fraction_set[MRG32] = (double)tot / n;
                thrust::fill(d_results.begin(), d_results.end(), 0u);
            }
        }

        print_header(n, BITMASK);
        print_row(PHILOX, fraction_set[PHILOX], philox_states.size_bytes(),
                  init_timings[PHILOX] / (double)(n_iter),
                  rand_timings[PHILOX] / (double)(n_iter));

        print_row(XORWOW, fraction_set[XORWOW], xorwow_states.size_bytes(),
                  init_timings[XORWOW] / (double)(n_iter),
                  rand_timings[XORWOW] / (double)(n_iter));

        print_row(MRG32, fraction_set[MRG32], mrg32_states.size_bytes(),
                  init_timings[MRG32] / (double)(n_iter),
                  rand_timings[MRG32] / (double)(n_iter));

        print_footer();
    }

    return EXIT_SUCCESS;
}
