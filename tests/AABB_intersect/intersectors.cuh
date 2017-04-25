#pragma once

#include "ray.cuh"
#include "AABB.cuh"

#include "aila_laine_karras.cuh"
#include "eisemann.cuh"
#include "plucker.cuh"
#include "williams.cuh"

// Type traits; AuxillaryTraits<Intersector>::type is the return type of
// Intersector::prepare(const Ray&). Specializations below.
template <typename Intersector>
struct AuxillaryTraits;

struct Aila
{
    __host__ __device__ AilaRayAuxillary prepare(const Ray& in_ray) const
    {
        return aila_auxillary(in_ray);
    }

    // This method uses device-only functions. The lack of __host__ is not a
    // mistake.
    __device__ int intersect(const Ray& ray,
                             const AilaRayAuxillary& aux,
                             const AABB& box) const
    {
        return aila(ray, aux, box);
    }
};

struct Eisemann
{
    __host__ __device__ EisemannRayAuxillary prepare(const Ray& in_ray) const
    {
        return eisemann_auxillary(in_ray);
    }

    __host__ __device__ int intersect(const Ray& ray,
                                      const EisemannRayAuxillary& aux,
                                      const AABB& box) const
    {
        return eisemann(ray, aux, box);
    }
};

struct Plucker
{
    __host__ __device__ PluckerRayAuxillary prepare(const Ray& in_ray) const
    {
        return plucker_auxillary(in_ray);
    }

    __host__ __device__ int intersect(const Ray& ray,
                                      const PluckerRayAuxillary& aux,
                                      const AABB& box) const
    {
        return plucker(ray, aux, box);
    }
};

struct Williams
{
    __host__ __device__ WilliamsRayAuxillary prepare(const Ray& in_ray) const
    {
        return williams_auxillary(in_ray);
    }

    __host__ __device__ int intersect(const Ray& ray,
                                      const WilliamsRayAuxillary& aux,
                                      const AABB& box) const
    {
        return williams(ray, aux, box);
    }
};

struct Williams_noif
{
     __host__ __device__ WilliamsRayAuxillary prepare(const Ray& in_ray) const
    {
        return williams_auxillary(in_ray);
    }

    __host__ __device__ int intersect(const Ray& ray,
                                      const WilliamsRayAuxillary& aux,
                                      const AABB& box) const
    {
        return williams_noif(ray, aux, box);
    }
};


template <>
struct AuxillaryTraits<Aila>
{
    typedef AilaRayAuxillary type;
};

template <>
struct AuxillaryTraits<Eisemann>
{
    typedef EisemannRayAuxillary type;
};

template <>
struct AuxillaryTraits<Plucker>
{
    typedef PluckerRayAuxillary type;
};

template <>
struct AuxillaryTraits<Williams>
{
    typedef WilliamsRayAuxillary type;
};

template <>
struct AuxillaryTraits<Williams_noif>
{
    typedef WilliamsRayAuxillary type;
};
