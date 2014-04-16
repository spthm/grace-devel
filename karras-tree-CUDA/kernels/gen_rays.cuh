#pragma once

#include <thrust/device_vector.h>

#include "../utils.cuh"

namespace grace {

template <typename Float4>
void square_grid_rays_z(thrust::device_vector<Ray>& d_rays,
                        const thrust::device_vector<Float4>& d_spheres,
                        const unsigned int N_rays_side)
{
    float min_x, max_x;
    min_max_x(&min_x, &max_x, d_spheres);

    float min_y, max_y;
    min_max_y(&min_y, &max_y, d_spheres);

    float min_z, max_z;
    min_max_z(&min_z, &max_z, d_spheres);

    float min_r, max_r;
    min_max_w(&min_r, &max_r, d_spheres);

    gpu::gen_grid_rays(d_rays, N_rays_side)
}

}
