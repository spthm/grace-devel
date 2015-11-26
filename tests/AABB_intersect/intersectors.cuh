#pragma once

#include "ray.cuh"
#include "AABB.cuh"

#include "aila_laine_karras.cuh"
#include "eisemann.cuh"
#include "plucker.cuh"
#include "williams.cuh"

struct Aila
{
    __host__ __device__ Ray prepare(const Ray& in_ray) const
    {
        return compute_ray_invd(in_ray);
    }

    __device__ int intersect(const Ray& ray, const AABB& box) const
    {
        return aila_laine_karras(ray, box);
    }
};

struct Eisemann
{
    __host__ __device__ Ray prepare(const Ray& in_ray) const
    {
        Ray ray = in_ray;
        ray = compute_ray_class(in_ray);
        ray = compute_ray_slope(ray);
        return ray;
    }

    __host__ __device__ int intersect(const Ray& ray, const AABB& box) const
    {
        return eisemann(ray, box);
    }
};

struct Plucker
{
    __host__ __device__ Ray prepare(const Ray& in_ray) const
    {
        return compute_ray_class(in_ray);
    }

    __host__ __device__ int intersect(const Ray& ray, const AABB& box) const
    {
        return plucker(ray, box);
    }
};

struct Williams
{
    __host__ __device__ Ray prepare(const Ray& in_ray) const
    {
        return compute_ray_invd(in_ray);
    }

    __host__ __device__ int intersect(const Ray& ray, const AABB& box) const
    {
        return williams(ray, box);
    }
};

struct Williams_noif
{
    __host__ __device__ Ray prepare(const Ray& in_ray) const
    {
        return compute_ray_invd(in_ray);
    }

    __host__ __device__ int intersect(const Ray& ray, const AABB& box) const
    {
        return williams_noif(ray, box);
    }
};
