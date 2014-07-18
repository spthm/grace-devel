#pragma once

#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "morton.cuh"
#include "sort.cuh"
#include "../kernel_config.h"
#include "../ray.h"
#include "../utils.cuh"

namespace grace {

namespace gpu {

__global__ void init_QRNG(curandStateSobol32_t *const qrng_states,
                          curandDirectionVectors32_t *const qrng_directions,
                          const unsigned int N_per_thread)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Initialise the Q-RNG
    // We may have that max(tid) < N_rays. For performance reasons, we only
    // only generate max(tid) states, and they are reused.  Each state's
    // initializing offset is such that it will not be 'caught' by the state
    // preceding it when states are reused.
    // Additionally, we +2 to the offset since the second value generated =~ -0
    // (for all three dimensions).
    curand_init(qrng_directions[0], (N_per_thread*tid)+2,
                &qrng_states[tid+0*stride]); // x
    curand_init(qrng_directions[1], (N_per_thread*tid)+2,
                &qrng_states[tid+1*stride]); // y
    curand_init(qrng_directions[2], (N_per_thread*tid)+2,
                &qrng_states[tid+2*stride]); // z
}

/* N normally distributed values (mean 0, fixed variance) normalized
 * to one gives us a uniform distribution on the unit N-dimensional
 * hypersphere. See e.g. Wolfram "[Hyper]Sphere Point Picking" and
 * http://www.math.niu.edu/~rusin/known-math/96/sph.rand
 *
 * We use a low-discrepancy quasi-RNG sequence (Sobol) as our
 * generator, and curand_normal uses the inverse CDF, preserving this
 * characteristic (unlike many other methods, which mix uniform input).
 * See http://stats.stackexchange.com/questions/27450
 */
template <typename Float4>
__global__ void gen_uniform_rays(Ray* rays,
                                 unsigned int* keys,
                                 const Float4 ol, // ox, oy, oz, length.
                                 curandStateSobol32_t *const qrng_states,
                                 const size_t N_rays)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    curandStateSobol32_t x_state = qrng_states[tid+0*stride];
    curandStateSobol32_t y_state = qrng_states[tid+1*stride];
    curandStateSobol32_t z_state = qrng_states[tid+2*stride];

    while (tid < N_rays)
    {
        float dx, dy, dz, invR;
        Ray ray;

        dx = curand_normal(&x_state);
        dy = curand_normal(&y_state);
        dz = curand_normal(&z_state);
        invR = 1.f / sqrt(dx*dx + dy*dy + dz*dz);

        ray.dx = dx*invR;
        ray.dy = dy*invR;
        ray.dz = dz*invR;

        // morton_key requires *floats* in (0, 1) for 30-bit keys.
        keys[tid] = morton_key((ray.dx+1)/2.f, (ray.dy+1)/2.f, (ray.dz+1)/2.f);

        ray.ox = ol.x;
        ray.oy = ol.y;
        ray.oz = ol.z;
        ray.length = ol.w;

        rays[tid] = ray;
        tid += stride;
    }
}

} // namespace gpu

template <typename Float>
void uniform_random_rays(thrust::device_vector<Ray>& d_rays,
                         const Float ox,
                         const Float oy,
                         const Float oz,
                         const Float length)
{
    size_t N_rays = d_rays.size();
    float4 origin = make_float4(ox, oy, oz, length);

    // We launch exactly enough blocks to fill the hardware for both of
    // these kernels, or fewer if there are not sufficient rays).
    // It is much more efficient to reuse qrng states than to generate new ones.
    // On Tesla M2090, there are 16MPs, so se have a maxiumum of 16 blocks.
    // On GTX 670, there are 7MPs, so we have a maximum of 7 blocks.
    int blocks = min(16, (int) ((N_rays + RAYS_THREADS_PER_BLOCK-1)
                                        / RAYS_THREADS_PER_BLOCK));

    // Upper bound on number of times curand_normal will be called by each
    // of the blocks * RAYS_THREADS_PER_BLOCK threads.
    unsigned int N_per_thread = (N_rays + blocks * RAYS_THREADS_PER_BLOCK - 1)
                                / (blocks * RAYS_THREADS_PER_BLOCK);

    // Allocate space for Q-RNG states and the three direction vectors.
    curandStateSobol32_t *d_qrng_states;
    cudaMalloc((void**)&d_qrng_states,
               3*blocks*RAYS_THREADS_PER_BLOCK * sizeof(curandStateSobol32_t));

    curandDirectionVectors32_t *d_qrng_directions;
    cudaMalloc((void**)&d_qrng_directions,
               3*sizeof(curandDirectionVectors32_t));

    // Generate Q-RNG 'direction vectors' on host, and copy to device.
    curandDirectionVectors32_t *qrng_directions;
    curandGetDirectionVectors32(&qrng_directions,
                                CURAND_DIRECTION_VECTORS_32_JOEKUO6);
    cudaMemcpy(d_qrng_directions, qrng_directions,
               3*sizeof(curandDirectionVectors32_t),
               cudaMemcpyHostToDevice);

    // Initialize the Q-RNG states.
    gpu::init_QRNG<<<blocks, RAYS_THREADS_PER_BLOCK>>>(d_qrng_states,
                                                       d_qrng_directions,
                                                       N_per_thread);

    thrust::device_vector<unsigned int> d_keys(N_rays);
    gpu::gen_uniform_rays<<<blocks, RAYS_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        origin,
        d_qrng_states,
        N_rays);

    cudaFree(d_qrng_states);
    cudaFree(d_qrng_directions);

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_rays.begin());
}

// template <typename Float4>
// void square_grid_rays_z(thrust::device_vector<Ray>& d_rays,
//                         const thrust::device_vector<Float4>& d_spheres,
//                         const unsigned int N_rays_side)
// {
//     float min_x, max_x;
//     min_max_x(&min_x, &max_x, d_spheres);

//     float min_y, max_y;
//     min_max_y(&min_y, &max_y, d_spheres);

//     float min_z, max_z;
//     min_max_z(&min_z, &max_z, d_spheres);

//     float min_r, max_r;
//     min_max_w(&min_r, &max_r, d_spheres);

//     gpu::gen_grid_rays(thrust::raw_pointer_cast(d_rays.data()),
//                        N_rays_side);
// }

} // namespace grace
