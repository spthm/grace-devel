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

// Binary encoding with P = 1, M = 0.
enum Octants {
    PPP = 7,
    PPM = 6,
    PMP = 5,
    PMM = 4,
    MPP = 3,
    MPM = 2,
    MMP = 1,
    MMM = 0
};

namespace gpu {

__global__ void init_PRNG_kernel(
    curandState* const prng_states,
    const unsigned long long seed,
    const int N)
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

/* N normally distributed values (mean 0, fixed variance) normalized
 * to one gives us a uniform distribution on the unit N-dimensional
 * hypersphere. See e.g. Wolfram "[Hyper]Sphere Point Picking" and
 * http://www.math.niu.edu/~rusin/known-math/96/sph.rand
 */
template <typename Real4, typename KeyType>
__global__ void gen_uniform_rays(
    const curandState* const prng_states,
    const Real4 ol, // ox, oy, oz, length.
    Ray* const rays,
    KeyType* const keys,
    const size_t N_rays)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state = prng_states[tid];

    while (tid < N_rays)
    {
        float dx, dy, dz, invR;
        Ray ray;

        dx = curand_normal(&state);
        dy = curand_normal(&state);
        dz = curand_normal(&state);
        #if __CUDACC_VER_MAJOR__ < 7
        invR = rsqrt(dx*dx + dy*dy + dz*dz);
        #else
        invR = rnorm3d(dx, dy, dz);
        #endif

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


template <typename Real4, typename KeyType>
__global__ void gen_uniform_rays_single_octant(
    const curandState* const prng_states,
    const Real4 ol, // ox, oy, oz, length.
    Ray* const rays,
    KeyType* const keys,
    const size_t N_rays,
    const enum Octants octant)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state = prng_states[tid];

    int3 sign;
    sign.x = (octant & 0x4) ? 1 : -1;
    sign.y = (octant & 0x2) ? 1 : -1;
    sign.z = (octant & 0x1) ? 1 : -1;

    while (tid < N_rays)
    {
        float dx, dy, dz, invR;
        Ray ray;

        dx = sign.x * fabsf(curand_normal(&state));
        dy = sign.y * fabsf(curand_normal(&state));
        dz = sign.z * fabsf(curand_normal(&state));
        #if __CUDACC_VER_MAJOR__ < 7
        invR = rsqrt(dx*dx + dy*dy + dz*dz);
        #else
        invR = rnorm3d(dx, dy, dz);
        #endif

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

// We launch only enough blocks to ~fill the hardware, minimizing the number
// of threads which must call the expensive curand_init().
//
// Up to max_states will be generated, and the number of states will be an
// integer multiple of factor.
//
// If max_states is not itself an integer multiple of factor, then the
// maximum number of states which may be returned is the smallest number greater
// than max_states which is a multiple of factor.
//
// The number of states is returned in N_states.
//
// Reusing the same state to produce multiple, different random numbers should
// not lead to any (significant) correlation effects.
GRACE_HOST void init_PRNG(
    const int max_states,
    const int factor,
    const unsigned long long seed,
    curandState** d_prng_states,
    int* N_states)
{
    const int block_size = 128; // For init_PRNG_kernel
    int device_ID, N_SMs;
    cudaDeviceProp prop;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_ID));
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_ID));
    N_SMs = prop.multiProcessorCount;

    // Ideally we would have the smaller of:
    //     max_states
    //     3 * N_SMs * block_size,
    // but this must be rounded up to a multiple of factor.
    int N = std::min(3 * N_SMs * block_size, max_states);
    N = factor * (N + factor - 1) / factor;
    *N_states = N;

    int num_blocks = (N + block_size - 1) / block_size;

    cudaError_t cuerr = cudaMalloc(
        (void**)d_prng_states,
        N * sizeof(curandState));
    GRACE_CUDA_CHECK(cuerr);

    // Initialize the P-RNG states.
    gpu::init_PRNG_kernel<<<num_blocks, block_size>>>(
        *d_prng_states,
        seed,
        N);
    GRACE_KERNEL_CHECK();
}

template <typename Real>
GRACE_HOST void uniform_random_rays(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const unsigned long long seed = 1234)
{
    float4 origin = make_float4(ox, oy, oz, length);

    curandState* d_prng_states;
    int N_states;
    init_PRNG(N_rays, RAYS_THREADS_PER_BLOCK, seed, &d_prng_states, &N_states);

    // init_PRNG guarantees N_states is a multiple of RAYS_THREADS_PER_BLOCK.
    int num_blocks = N_states / RAYS_THREADS_PER_BLOCK;

    thrust::device_vector<uinteger32> d_keys(N_rays);
    gpu::gen_uniform_rays<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        d_prng_states,
        origin,
        d_rays_ptr,
        thrust::raw_pointer_cast(d_keys.data()),
        N_rays);
    GRACE_KERNEL_CHECK();

    GRACE_CUDA_CHECK(cudaFree(d_prng_states));

    thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                        thrust::device_ptr<Ray>(d_rays_ptr));
}

template <typename Real>
GRACE_HOST void uniform_random_rays(
    thrust::device_vector<Ray>& d_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const unsigned long long seed = 1234)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    size_t N_rays = d_rays.size();

    uniform_random_rays(d_rays_ptr, N_rays, ox, oy, oz, length, seed);
}

template <typename Real>
GRACE_HOST void uniform_random_rays_single_octant(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const unsigned long long seed = 1234,
    const enum Octants octant = PPP)
{
    float4 origin = make_float4(ox, oy, oz, length);

    curandState* d_prng_states;
    int N_states;
    init_PRNG(N_rays, RAYS_THREADS_PER_BLOCK, seed, &d_prng_states, &N_states);

    // init_PRNG guarantees N_states is a multiple of RAYS_THREADS_PER_BLOCK.
    int num_blocks = N_states / RAYS_THREADS_PER_BLOCK;

    thrust::device_vector<uinteger32> d_keys(N_rays);
    gpu::gen_uniform_rays_single_octant<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        d_prng_states,
        origin,
        d_rays_ptr,
        thrust::raw_pointer_cast(d_keys.data()),
        N_rays,
        octant);
    GRACE_KERNEL_CHECK();

    GRACE_CUDA_CHECK(cudaFree(d_prng_states));

    thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                        thrust::device_ptr<Ray>(d_rays_ptr));
}

template <typename Real>
GRACE_HOST void uniform_random_rays_single_octant(
    thrust::device_vector<Ray>& d_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const unsigned long long seed = 1234,
    const enum Octants octant = PPP)
{
    Ray* const d_rays_ptr = thrust::raw_pointer_cast(d_rays.data());
    size_t N_rays = d_rays.size();

    uniform_random_rays_single_octant(d_rays_ptr, N_rays, ox, oy, oz, length,
                                      seed, octant);

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

//     random::gen_grid_rays(thrust::raw_pointer_cast(d_rays.data()),
//                        N_rays_side);
// }

} // namespace grace
