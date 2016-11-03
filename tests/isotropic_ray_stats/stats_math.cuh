#pragma once

#include "grace/ray.h"

#include <thrust/functional.h>

const double PI = 3.141592653589793;

struct
raydir_add : public thrust::binary_function<grace::Ray, grace::Ray, grace::Ray>
{
    __host__ __device__
    grace::Ray operator()(grace::Ray lhs, grace::Ray rhs) const
    {
        grace::Ray ray;
        ray.dx = lhs.dx + rhs.dx;
        ray.dy = lhs.dy + rhs.dy;
        ray.dz = lhs.dz + rhs.dz;
        return ray;
    }
};
