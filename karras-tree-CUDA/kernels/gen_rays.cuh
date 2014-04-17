#pragma once

#include <thrust/device_vector.h>

#include "sort.cuh"
#include "../utils.cuh"

namespace grace {

namespace gpu {

template <typename Float4, typename Float>
__global__ void gen_uniform_rays(Float4* ray_dirs,
                                 Float4* ray_origs,
                                 const Float4 origin,
                                 const Float length,
                                 const unsigned int N_rays)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Ray ray;

    while (tid < N_rays) {
        // Do some cuRAND stuff to get a random number.
        // If we generate normally distributed values using cuRAND for x, y, z,
        // that automatically gives us a uniform distribution on the unit sphere
        // see http://www.math.niu.edu/~rusin/known-math/96/sph.rand

        float invR = 1. / sqrt(dx*dx + dy*dy + dz*dz);

        ray_dirs.x = dx * invR;
        ray_dirs.y = dy * invR;
        ray_dirs.z = dz * invR;
        ray_origs.x = origin.x;
        ray_origs.y = origin.y;
        ray_origs.z = origin.z;
        ray_origs.w = length;

        int dclass = 0;
        if (dx >= 0)
            ray.dclass += 1;
        if (dy >= 0)
            ray.dclass += 2;
        if (dz >= 0)
            ray.dclass += 4;

        ray_dirs.w = __int_as_float(dclass);

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
