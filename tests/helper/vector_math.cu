#include <math.h>

#include "vector_math.cuh"

__host__ __device__ double magnitude(float3 v)
{
    double v2 = (double)v.x * v.x + (double)v.y * v.y + (double)v.z * v.z;
    return sqrt(v2);
}

__host__ __device__ double magnitude(double3 v)
{
    double v2 = v.x * v.x + v.y * v.y + v.z * v.z;
    return sqrt(v2);
}

__host__ __device__ float3 normalize(float3 v)
{
    return v / magnitude(v);
}

__host__ __device__ double3 normalize(double3 v)
{
     return v / magnitude(v);
}

__host__ __device__ double dot_product(float3 a, float3 b)
{
    double x = (double)a.x * b.x;
    double y = (double)a.y * b.y;
    double z = (double)a.z * b.z;
    return x + y + z;
}

__host__ __device__ double dot_product(double3 a, double3 b)
{
    double x = a.x * b.x;
    double y = a.y * b.y;
    double z = a.z * b.z;
    return x + y + z;
}

__host__ __device__ float3 cross_product(float3 a, float3 b)
{
    float3 res;
    res.x = (double)a.y * b.z - (double)a.z * b.y;
    res.y = (double)a.z * b.x - (double)a.x * b.z;
    res.z = (double)a.x * b.y - (double)a.y * b.x;
    return res;
}

__host__ __device__ double3 cross_product(double3 a, double3 b)
{
    double3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

__host__ __device__ double angular_separation(float3 p, float3 q)
{
    // This form is well conditioned when the angle is 0 or pi; acos is not.
    return atan2(magnitude(cross_product(p, q)), dot_product(p, q));
}

__host__ __device__ double angular_separation(double3 p, double3 q)
{
    return atan2(magnitude(cross_product(p, q)), dot_product(p, q));
}

__host__ __device__ double great_circle_distance(float3 p, float3 q, float R)
{
    return R * angular_separation(p, q);
}

__host__ __device__ double great_circle_distance(double3 p, double3 q, double R)
{
    return R * angular_separation(p, q);
}

__host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x,
                       a.y + b.y,
                       a.z + b.z);
}

__host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x,
                       a.y - b.y,
                       a.z - b.z);
}

__host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x,
                       a.y * b.y,
                       a.z * b.z);
}

__host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x,
                       a.y / b.y,
                       a.z / b.z);
}

__host__ __device__ float3 operator+(float s, float3 v)
{
    return make_float3(s + v.x,
                       s + v.y,
                       s + v.z);
}

__host__ __device__ float3 operator-(float s, float3 v)
{
    return make_float3(s - v.x,
                       s - v.y,
                       s - v.z);
}

__host__ __device__ float3 operator*(float s, float3 v)
{
    return make_float3(s * v.x,
                       s * v.y,
                       s * v.z);
}

__host__ __device__ float3 operator/(float s, float3 v)
{
    return make_float3(s / v.x,
                       s / v.y,
                       s / v.z);
}

__host__ __device__ float3 operator+(float3 v, float s)
{
    return make_float3(v.x + s,
                       v.y + s,
                       v.z + s);
}

__host__ __device__ float3 operator-(float3 v, float s)
{
    return make_float3(v.x - s,
                       v.y - s,
                       v.z - s);
}

__host__ __device__ float3 operator*(float3 v, float s)
{
    return make_float3(v.x * s,
                       v.y * s,
                       v.z * s);
}

__host__ __device__ float3 operator/(float3 v, float s)
{
    return make_float3(v.x / s,
                       v.y / s,
                       v.z / s);
}

__host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x,
                        a.y + b.y,
                        a.z + b.z);
}

__host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x,
                        a.y - b.y,
                        a.z - b.z);
}

__host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x,
                        a.y * b.y,
                        a.z * b.z);
}

__host__ __device__ double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x,
                        a.y / b.y,
                        a.z / b.z);
}

__host__ __device__ double3 operator+(double s, double3 v)
{
    return make_double3(s + v.x,
                        s + v.y,
                        s + v.z);
}

__host__ __device__ double3 operator-(double s, double3 v)
{
    return make_double3(s - v.x,
                        s - v.y,
                        s - v.z);
}

__host__ __device__ double3 operator*(double s, double3 v)
{
    return make_double3(s * v.x,
                        s * v.y,
                        s * v.z);
}

__host__ __device__ double3 operator/(double s, double3 v)
{
    return make_double3(s / v.x,
                        s / v.y,
                        s / v.z);
}

__host__ __device__ double3 operator+(double3 v, double s)
{
    return make_double3(v.x + s,
                        v.y + s,
                        v.z + s);
}

__host__ __device__ double3 operator-(double3 v, double s)
{
    return make_double3(v.x - s,
                        v.y - s,
                        v.z - s);
}

__host__ __device__ double3 operator*(double3 v, double s)
{
    return make_double3(v.x * s,
                        v.y * s,
                        v.z * s);
}

__host__ __device__ double3 operator/(double3 v, double s)
{
    return make_double3(v.x / s,
                        v.y / s,
                        v.z / s);
}
