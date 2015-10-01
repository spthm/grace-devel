#pragma once

#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "sort.cuh"
#include "../device/morton.cuh"
#include "../error.h"
#include "../kernel_config.h"
#include "../ray.h"
#include "../types.h"
#include "../utils.cuh"

namespace grace {

namespace gpu {

__global__ void init_PRNG(
    curandState* const prng_states,
    const unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialise the P-RNG
    // We generate one RNG state per ray.
    // Following the cuRAND documentation, each thread receives the same seed
    // value, no offset, and a *different* sequence value.
    // This should prevent any correlations.
    curand_init(seed, tid, 0, &prng_states[tid]);
}

/* N normally distributed values (mean 0, fixed variance) normalized
 * to one gives us a uniform distribution on the unit N-dimensional
 * hypersphere. See e.g. Wolfram "[Hyper]Sphere Point Picking" and
 * http://www.math.niu.edu/~rusin/known-math/96/sph.rand
 */
template <typename Float4>
__global__ void gen_uniform_rays(
    curandState* const prng_states,
    const Float4 ol, // ox, oy, oz, length.
    Ray* rays,
    unsigned int* keys,
    const size_t N_rays)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state = prng_states[tid];

    while (tid < N_rays)
    {
        float dx, dy, dz, invR;
        Ray ray;

        dx = curand_normal(&state);
        dy = curand_normal(&state);
        dz = curand_normal(&state);
        invR = 1.f / sqrt(dx*dx + dy*dy + dz*dz);

        ray.dx = dx * invR;
        ray.dy = dy * invR;
        ray.dz = dz * invR;

        // morton_key requires *floats* in (0, 1) for 30-bit keys.
        keys[tid] = morton::morton_key((ray.dx+1)/2.f,
                                       (ray.dy+1)/2.f,
                                       (ray.dz+1)/2.f);

        ray.ox = ol.x;
        ray.oy = ol.y;
        ray.oz = ol.z;
        ray.length = ol.w;

        rays[tid] = ray;

        tid += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

template <typename Float>
GRACE_HOST void uniform_random_rays(
    thrust::device_vector<Ray>& d_rays,
    const Float ox,
    const Float oy,
    const Float oz,
    const Float length,
    const unsigned long long seed = 1234)
{
    cudaError_t cuerr;

    size_t N_rays = d_rays.size();
    float4 origin = make_float4(ox, oy, oz, length);

    // We launch only enough blocks to ~fill the hardware, minimizing the number
    // of threads which must call the expensive curand_init().
    // Reusing the same generator for multiple rays (as will occur whenever
    // N_rays > init_blocks * RAYS_THREADS_PER_BLOCK) should not lead to any
    // (significant) correlation effects.
    int device_ID, N_SMs;
    cudaDeviceProp prop;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_ID));
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_ID));
    N_SMs = prop.multiProcessorCount;

    int num_blocks =
        std::min(3 * N_SMs, static_cast<int>((N_rays + RAYS_THREADS_PER_BLOCK-1)
                                             / RAYS_THREADS_PER_BLOCK));

    // Allocate space for Q-RNG states and the three direction vectors.
    curandState *d_prng_states;
    cuerr = cudaMalloc(
        (void**)&d_prng_states,
        num_blocks * RAYS_THREADS_PER_BLOCK * sizeof(curandState));
    GRACE_CUDA_CHECK(cuerr);

    // Initialize the Q-RNG states.
    gpu::init_PRNG<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(d_prng_states,
                                                           seed);
    GRACE_KERNEL_CHECK();

    thrust::device_vector<unsigned int> d_keys(N_rays);
    gpu::gen_uniform_rays<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        d_prng_states,
        origin,
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        N_rays);
    GRACE_KERNEL_CHECK();

    GRACE_CUDA_CHECK(cudaFree(d_prng_states));

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_rays.begin());
}

// template <typename Float4>
// GRACE_HOST void square_grid_rays_z(
//     thrust::device_vector<Ray>& d_rays,
//     const thrust::device_vector<Float4>& d_spheres,
//     const unsigned int N_rays_side)
// {
//     float min_x, max_x;
//     min_max_x(d_spheres, &min_x, &max_x);

//     float min_y, max_y;
//     min_max_y(d_spheres, &min_y, &max_y);

//     float min_z, max_z;
//     min_max_z(d_spheres, &min_z, &max_z);

//     float min_r, max_r;
//     min_max_w(d_spheres, &min_r, &max_r);

//     gpu::gen_grid_rays(thrust::raw_pointer_cast(d_rays.data()),
//                        N_rays_side);
// }

} // namespace grace
