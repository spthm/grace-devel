#ifndef AABB_H
#define AABB_H

struct AABB
{
    __host__ __device__ AABB() {}
    __host__ __device__ AABB(const float3 &c, const float3 &hs)
        : center(c), hsize(hs) {}

    // order is important, raytrace.cu depends on it
    float3 center;
    float3 hsize;
};

#endif
