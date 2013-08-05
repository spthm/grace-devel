#ifndef RAYTRACE_H
#define RAYTRACE_H

#include "types.h"

struct linear_8bvh_node;

__host__ void cuda_trace(float3 U, float3 V, float3 W, 
                         float3 invU, float3 invV, float3 invW,
                         float3 eye,
                         const Mesh &mesh, 
                         const std::vector<linear_8bvh_node> &bvh,
                         size_t bvh_height,
                         cudaGraphicsResource *output,
                         int output_width, int output_height,
                         bool recalc_rays=false, bool reload_model=false);

#endif
