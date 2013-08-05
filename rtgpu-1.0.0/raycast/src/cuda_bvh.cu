#include "cuda_bvh.h"
#include "bvh.h"

void convert_to_soa(bvh_soa &soa, const std::vector<linear_8bvh_node> &bvh)
{
    soa.aabb1.resize(bvh.size());
    cudaMemcpy2D(soa.aabb1, sizeof(float4), 
                 &bvh[0].aabb.center.x, sizeof(bvh[0]), 
                 sizeof(float4), bvh.size(), cudaMemcpyHostToDevice);
    check_cuda_error("Copying node_aabb1 to device");

    //---------------------------------------------

    soa.aabb2.resize(bvh.size());
    cudaMemcpy2D(soa.aabb2, sizeof(float2), 
                 &bvh[0].aabb.hsize.y, sizeof(bvh[0]), 
                 sizeof(float2), bvh.size(), cudaMemcpyHostToDevice);
    check_cuda_error("Copying node_aabb2 to device");

    //---------------------------------------------

    soa.children_info.resize(bvh.size());
    cudaMemcpy2D(soa.children_info, sizeof(unsigned), 
                 &bvh[0].children_info, sizeof(bvh[0]), 
                 sizeof(unsigned), bvh.size(), cudaMemcpyHostToDevice);
    check_cuda_error("Copying node_children_info to device");

    //---------------------------------------------

    soa.prim_info.resize(bvh.size());
    cudaMemcpy2D(soa.prim_info, sizeof(unsigned), 
                 &bvh[0].prim_info, sizeof(bvh[0]), 
                 sizeof(unsigned), bvh.size(), cudaMemcpyHostToDevice);
    check_cuda_error("Copying node_prim_info to device");
}
