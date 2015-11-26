#pragma once

#include <thrust/host_vector.h>

struct Ray
{
    float dx, dy, dz;
    float ox, oy, oz;
    float length;

    // Inverse of ray's direction components; only required for Williams and
    // Aila-Laine-Karras methods.
    float invdx, invdy, invdz;

    // Ray direction octant; only required for Eisemann and Plucker methods.
    unsigned int dclass;

    // Ray slope data; only required for Eisemann method.
    float xbyy, ybyx, ybyz, zbyy, xbyz, zbyx;
    float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
};

enum DIR_CLASS { MMM, PMM, MPM, PPM, MMP, PMP, MPP, PPP };
enum { MISS = 0, HIT = 1 };

__host__ __device__ Ray compute_ray_invd(const Ray&);
__host__ __device__ Ray compute_ray_class(const Ray&);
__host__ __device__ Ray compute_ray_slope(const Ray&);

void isotropic_rays(thrust::host_vector<Ray>&, float ox, float oy, float oz,
                    float length);
