#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "dvector.h"
#include "aabb.h"

struct bvh_soa
{
    dvector<float4> aabb1; // center.x, center.y, center.z, hsize.x
    dvector<float2> aabb2; // hsize.y, hsize.z
    dvector<unsigned> children_info,
                      prim_info;
};

inline __device__ AABB make_aabb(const float4 &aabb1, const float2 &aabb2)
{
    AABB aabb;
    aabb.center = make_float3(aabb1.x, aabb1.y, aabb1.z);
    aabb.hsize = make_float3(aabb1.w, aabb2.x, aabb2.y);
    return aabb;
}

void convert_to_soa(bvh_soa &soa, const std::vector<linear_8bvh_node> &bvh);


#endif
