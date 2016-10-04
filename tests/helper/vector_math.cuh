#pragma once

#include "grace/ray.h"

#include <thrust/functional.h>

// All versions compute variables in double-precision, hence the return type
// is typically double.
__host__ __device__ double magnitude(float3 v);
__host__ __device__ double magnitude(double3 v);

__host__ __device__ double dot_product(float3 a, float3 b);
__host__ __device__ double dot_product(double3 a, double3 b);

__host__ __device__ float3 cross_product(float3 a, float3 b);
__host__ __device__ double3 cross_product(double3 a, double3 b);

__host__ __device__ double angular_separation(float3 p, float3 q);
__host__ __device__ double angular_separation(double3 p, double3 q);

__host__ __device__ double great_circle_distance(float3 p, float3 q,
                                                 float R = 1.0f);
__host__ __device__ double great_circle_distance(double3 p, double3 q,
                                                 double R = 1.0);

__host__ __device__ float3 operator+(float3 a, float3 b);
__host__ __device__ float3 operator-(float3 a, float3 b);
__host__ __device__ float3 operator*(float3 a, float3 b);
__host__ __device__ float3 operator/(float3 a, float3 b);

__host__ __device__ float3 operator+(float s, float3 v);
__host__ __device__ float3 operator-(float s, float3 v);
__host__ __device__ float3 operator*(float s, float3 v);
__host__ __device__ float3 operator/(float s, float3 v);

__host__ __device__ float3 operator+(float3 v, float s);
__host__ __device__ float3 operator-(float3 v, float s);
__host__ __device__ float3 operator*(float3 v, float s);
__host__ __device__ float3 operator/(float3 v, float s);

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
