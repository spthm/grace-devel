#pragma once

#include <cuda.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>

#include "sort.cuh"
#include "../ray.h"
#include "../utils.cuh"

namespace grace {

namespace gpu {

__global__ void initQRNG(curandStateSobol32_t *const qrng_states,
                         curandDirectionVectors32_t *const qrng_directions)
{
    unsigned int tid = threadIdx. + blockIdx.x * blockDim.x;

    // Initialise the Q-RNG
    curand_init(qrng_directions[0], tid, &qrng_states[3*tid+0]); // x
    curand_init(qrng_directions[1], tid, &qrng_states[3*tid+1]); // y
    curand_init(qrng_directions[2], tid, &qrng_states[3*tid+2]); // z
    }
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
__global__ void gen_uniform_rays(Ray* ray_dirs,
                                 const Float4 origin, // ox, oy, oz, length.
                                 curandStateSobol32_t *const qrng_states,
                                 const unsigned int N_rays)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandStateSobol32_t x_state = qrng_states[3*tid+0];
    curandStateSobol32_t y_state = qrng_states[3*tid+1];
    curandStateSobol32_t z_state = qrng_states[3*tid+2];

    float dx, dy, dz;
    Ray ray;

    while (tid < N_rays) {

        dx = curand_normal(&x_state);
        dy = curand_normal(&y_state);
        dz = curand_normal(&z_state);

        float invR = 1. / sqrt(dx*dx + dy*dy + dz*dz);

        ray.dx = dx * invR;
        ray.dy = dy * invR;
        ray.dz = dz * invR;
        ray.ox = origin.x;
        ray.oy = origin.y;
        ray.oz = origin.z;
        ray.length = origin.w;

        unsigned int dclass = 0;
        if (dx >= 0)
            ray.dclass += 1;
        if (dy >= 0)
            ray.dclass += 2;
        if (dz >= 0)
            ray.dclass += 4;
        ray.dclass = dclass;

        rays[tid] = ray;
        tid += blockDim.x * gridDim.x;
    }

} // namespace gpu

template <typename Float4, typename Float>
void uniform_random_rays(thrust::device_vector<Ray>& d_rays,
                         const Float4 origin,
                         const Float length,
                         const unsigned int N_rays)
{
    // Allocate space for Q-RNG states and the three direction vectors.
    curandStateSobol32_t *d_qrng_states;
    cudaMalloc((void**)&d_qrng_states,
               3*N_rays * sizeof(curandStateSobol32_t));

    curandDirectionVectors32_t *d_qrng_directions;
    cudaMalloc((void**)&d_qrng_directions,
               3*sizeof(curandDirectionVectors32_t));

    // Generate direction vectors on host, and copy to device.
    curandDirectionVectors32_t *qrng_directions;
    curandGetDirectionVectors32(&qrng_directions,
                                CURAND_DIRECTION_VECTORS_32_JOEKUO6);
    cudaMemcpy(d_qrng_directions, qrng_directions,
               3*sizeof(curandDirectionVectors32_t),
               cudaMemcpyHostToDevice);


    gpu::initQRNG(d_qrng_states, d_qrng_directions);
    gpu::gen_uniform_rays(thrust::raw_pointer_cast(d_rays.dir.data()),
                          thrust::raw_pointer_cast(d_rays.orig.data()),
                          origin,
                          length,
                          N_rays);

    thrust::device_vector<unsigned int> d_keys(N_rays);
    morton_keys(d_keys, d_ray_dirs);
    grace::sort_by_key(d_keys, d_rays.dir, d_rays.orig);
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
