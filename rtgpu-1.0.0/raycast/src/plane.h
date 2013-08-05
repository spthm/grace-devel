#ifndef CUDA_PLANE_H
#define CUDA_PLANE_H

typedef float4 Plane;

inline __host__ __device__ 
float signed_distance(const Plane &plane, const float3 &pos)
{
    return dot(make_float3(plane.x,plane.y,plane.z), pos) + plane.w;
}

inline __host__ __device__ 
bool intersects(const AABB &aabb, const Plane &plane)
{
    float e = aabb.hsize.x * fabs(plane.x) + 
              aabb.hsize.y * fabs(plane.y) + 
              aabb.hsize.z * fabs(plane.z);

    float s = dot(aabb.center, make_float3(plane.x,plane.y,plane.z)) + plane.w;

    return s-e <= 0;
}

inline __host__ __device__ 
bool intersects(const AABB &aabb, const volatile Plane &plane)
{
    float e = aabb.hsize.x * fabs(plane.x) + 
              aabb.hsize.y * fabs(plane.y) + 
              aabb.hsize.z * fabs(plane.z);

    float s = dot(aabb.center, make_float3(plane.x,plane.y,plane.z)) + plane.w;

    return s-e <= 0;
}

#endif
