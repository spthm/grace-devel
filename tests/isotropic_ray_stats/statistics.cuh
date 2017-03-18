#pragma once

#include "grace/ray.h"

#include <thrust/device_vector.h>

template <typename Real2>
struct real2_add : public thrust::binary_function<Real2, Real2, Real2>
{
    __host__ __device__
    Real2 operator()(Real2 lhs, Real2 rhs) const
    {
        Real2 res;
        res.x = lhs.x + rhs.x;
        res.y = lhs.y + rhs.y;
        return res;
    }
};

float resultant_length_squared(const thrust::device_vector<grace::Ray>& rays);

void An_Gn_statistics(const thrust::device_vector<grace::Ray>& rays,
                      double* An, double* Gn);
