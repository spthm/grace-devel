#pragma once

#include "ray.cuh"

template <typename Aux>
__host__ __device__ void set_invd(const Ray& ray, Aux& aux)
{
    aux.invdx = 1.0f / ray.dx;
    aux.invdy = 1.0f / ray.dy;
    aux.invdz = 1.0f / ray.dz;
}

template <typename Aux>
__host__ __device__ void set_dclass(const Ray& ray, Aux& aux)
{
    aux.dclass = 0;
    if (ray.dx >= 0)
        aux.dclass += 1;
    if (ray.dy >= 0)
        aux.dclass += 2;
    if (ray.dz >= 0)
        aux.dclass += 4;
}

template <typename Aux>
__host__ __device__ void set_slope(const Ray& ray, Aux& aux)
{
    aux.xbyy = ray.dx / ray.dy;
    aux.ybyx = 1.0f / aux.xbyy;
    aux.ybyz = ray.dy / ray.dz;
    aux.zbyy = 1.0f / aux.ybyz;
    aux.xbyz = ray.dx / ray.dz;
    aux.zbyx = 1.0f / aux.xbyz;

    aux.c_xy = ray.oy - aux.ybyx * ray.ox;
    aux.c_xz = ray.oz - aux.zbyx * ray.ox;
    aux.c_yx = ray.ox - aux.xbyy * ray.oy;
    aux.c_yz = ray.oz - aux.zbyy * ray.oy;
    aux.c_zx = ray.ox - aux.xbyz * ray.oz;
    aux.c_zy = ray.oy - aux.ybyz * ray.oz;
}
