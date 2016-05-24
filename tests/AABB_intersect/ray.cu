#include "ray.cuh" // THIS
#include "grace/ray.h" // GRACE
#include "grace/cuda/gen_rays.cuh"

#include <thrust/device_vector.h>

__host__ __device__ Ray compute_ray_invd(const Ray& in_ray)
{
    Ray ray = in_ray;

    ray.invdx = 1.0f / ray.dx;
    ray.invdy = 1.0f / ray.dy;
    ray.invdz = 1.0f / ray.dz;

    return ray;
}

__host__ __device__ Ray compute_ray_class(const Ray& in_ray)
{
    Ray ray = in_ray;

    if (ray.dx >= 0)
        ray.dclass += 1;
    if (ray.dy >= 0)
        ray.dclass += 2;
    if (ray.dz >= 0)
        ray.dclass += 4;

    return ray;
}

__host__ __device__ Ray compute_ray_slope(const Ray& in_ray)
{
    Ray ray = in_ray;

    ray.xbyy = ray.dx / ray.dy;
    ray.ybyx = 1.0f / ray.xbyy;
    ray.ybyz = ray.dy / ray.dz;
    ray.zbyy = 1.0f / ray.ybyz;
    ray.xbyz = ray.dx / ray.dz;
    ray.zbyx = 1.0f / ray.xbyz;

    ray.c_xy = ray.oy - ray.ybyx*ray.ox;
    ray.c_xz = ray.oz - ray.zbyx*ray.ox;
    ray.c_yx = ray.ox - ray.xbyy*ray.oy;
    ray.c_yz = ray.oz - ray.zbyy*ray.oy;
    ray.c_zx = ray.ox - ray.xbyz*ray.oz;
    ray.c_zy = ray.oy - ray.ybyz*ray.oz;

    return ray;
}

void isotropic_rays(
    thrust::host_vector<Ray>& h_rays,
    float ox,
    float oy,
    float oz,
    float length)
{
    const size_t N_rays = h_rays.size();

    thrust::device_vector<grace::Ray> d_grace_rays(N_rays);
    grace::uniform_random_rays(
        thrust::raw_pointer_cast(d_grace_rays.data()),
        N_rays,
        ox, oy, ox, length);

    // Copy GRACE-type rays to the required ray type.
    thrust::host_vector<grace::Ray> h_grace_rays = d_grace_rays;
    for (size_t i = 0; i < N_rays; ++i)
    {
        h_rays[i].dx = h_grace_rays[i].dx;
        h_rays[i].dy = h_grace_rays[i].dy;
        h_rays[i].dz = h_grace_rays[i].dz;

        h_rays[i].ox = h_grace_rays[i].ox;
        h_rays[i].oy = h_grace_rays[i].oy;
        h_rays[i].oz = h_grace_rays[i].oz;

        h_rays[i].length = h_grace_rays[i].length;
    }
}
