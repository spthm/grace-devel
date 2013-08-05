#include <float.h>
#include <stdio.h>
#include <iostream>
#include <cassert>
#include "timer.h"
#include "symbol.h"
#include "packet_decomposition.h"
#include "dvector.h"
#include "util.h"
#include "aabb.h"
#include "bvh.h"
#include "ray.h"
#include "triangle.cu"
#include "cuda_bvh.h"
#include "light.h"

texture<float4> tex_bvh_node_aabb1;
texture<float2> tex_bvh_node_aabb2;
texture<unsigned> tex_bvh_node_prim_info;

texture<float4> tex_tri_xform;
texture<float4> tex_tri_normals;

inline __device__ AABB bvh_node_aabb(int idnode)
{
    return make_aabb(tex1Dfetch(tex_bvh_node_aabb1, idnode),
                     tex1Dfetch(tex_bvh_node_aabb2, idnode));
}

#define SCENE_EPSILON 0.01

#define PERSISTENT_WARPS 8
#define BATCH_SIZE 2

#define SHADOW_PERSISTENT_WARPS 8
#define SHADOW_BATCH_SIZE 5

__constant__ unsigned global_pool_warp_count;
__device__ unsigned global_pool_next_warp;

// primary local intersections
__global__ void primary_local_intersections(const unsigned *ray_warps,/*{{{*/
                                  const unsigned *ray_warp_size,
                                  const unsigned *warp_frustum,
                                  const unsigned *frustum_leaf_base,
                                  const unsigned *frustum_leaf_size,
                                  const unsigned *idleaves,
                                  const unsigned *ray_idx,
                                  float3 ray_ori, const float3 *ray_dirs,
                                  float3 *isect_positions,
                                  float3 *isect_normals,
                                  int *isect_indices,
                                  unsigned *pos_ray_warp)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ volatile unsigned next_warp_array[PERSISTENT_WARPS],
                                 warp_count_array[PERSISTENT_WARPS];

    volatile unsigned &idwarp = next_warp_array[ty],
                      &local_pool_warp_count = warp_count_array[ty];

    if(tx == 0)
        local_pool_warp_count = 0;

    while(true)
    {
        if(local_pool_warp_count == 0 && tx == 0)
        {
            idwarp = atomicAdd(&global_pool_next_warp,BATCH_SIZE);
            local_pool_warp_count = BATCH_SIZE;
        }

        if(idwarp >= global_pool_warp_count)
            return;

        unsigned warp_size = ray_warp_size[idwarp];

        float3 ray_dir;
        int idray;

        float3 N;
        float t = FLT_MAX;
        int idleaf_int;

        if(tx < warp_size)
        {
            idray = ray_idx[ray_warps[idwarp] + tx];
            ray_dir = ray_dirs[idray];
        }

        unsigned idfrustum = warp_frustum[idwarp]-1;

        int leaf_count = frustum_leaf_size[idfrustum],
            leaf_base = frustum_leaf_base[idfrustum] + leaf_count;

        // for all leaves
        for(int idrefleaf = -leaf_count; idrefleaf;) 
        {
#define IDLEAVES_PREFETCH 16
            __shared__ unsigned idleaves_cache[PERSISTENT_WARPS*IDLEAVES_PREFETCH];
            int bounds_count = min(IDLEAVES_PREFETCH,-idrefleaf);
            if(tx < bounds_count)
                idleaves_cache[ty*IDLEAVES_PREFETCH+tx] = idleaves[leaf_base+idrefleaf+tx];

            if(tx < warp_size)
            {
                for(int i=0; i<bounds_count; ++i)
                {
                    int idleaf = idleaves_cache[ty*IDLEAVES_PREFETCH+i];
                    // ray intersect leaf aabb?
                    if(intersects(ray_ori, 1.0f/ray_dir,0,t, 
                                  bvh_node_aabb(idleaf)))
                    {
                        unsigned prim_info = 
                            tex1Dfetch(tex_bvh_node_prim_info, idleaf);

                        unsigned prim_base = (prim_info>>8);
                        // for all primitives in this leaf,

                        for(int idtri = 0; idtri < (prim_info & 0xFF); ++idtri)
                        {
                            int idprim = (prim_base+idtri)*3;

                            if(intersect(ray_ori, ray_dir, SCENE_EPSILON, t,
                                tex_tri_xform, &tex_tri_normals, idprim,
                                &t, &N))
                           {
                               idleaf_int = idleaf;
                           }
                        }
                    }
                }
            }

            idrefleaf += bounds_count;
        }

        if(tx < warp_size)
        {
            int pos = pos_ray_warp[idwarp]+tx;

            if(t < FLT_MAX)
            {
                float3 ori = ray_ori + t*ray_dir;
#if 0

                ray_dir = make_float3(0,0,0) - ori;

                t = 1;

                AABB aabb = bvh_node_aabb(idleaf_int);

                intersects(ori, 1.0f/ray_dir,0,t, aabb, &t);
//                cuPrintf("%f\n",t);
                t += SCENE_EPSILON;

                isect_positions[pos] = ori + t*ray_dir;
#else
                isect_positions[pos] = ori;
#endif

                isect_normals[pos] = unit(N);
                isect_indices[pos] = idray;
            }
            else
                isect_indices[pos] = -1;
        }

        if(tx == 0)
        {
            ++idwarp;
            --local_pool_warp_count;
        }
    }
}/*}}}*/

__host__ void primary_local_intersections(
    dvector<float3> &d_isect_positions,
    dvector<float3> &d_isect_normals,
    dvector<int> &d_isect_rays_idx,

    const dvector<unsigned> &d_active_ray_packets,
    const dvector<unsigned> &d_active_ray_packet_sizes,
    const dvector<unsigned> &d_active_frustum_leaves,
    const dvector<unsigned> &d_active_frustum_leaf_sizes,
    const dvector<unsigned> &d_active_idleaves,
    const bvh_soa &d_bvh,

    const dvector<float4> &d_tri_xform,
    const dvector<float4> &d_tri_normals,

    float3 ray_ori, const dvector<float3> &d_rays_dir,
    const dvector<unsigned> &d_rays_idx)
{
    cudaBindTexture(NULL, tex_bvh_node_aabb1, d_bvh.aabb1);
    check_cuda_error("Binding aabb1 to texture");

    cudaBindTexture(NULL, tex_bvh_node_aabb2, d_bvh.aabb2);
    check_cuda_error("Binding aabb2 to texture");

    cudaBindTexture(NULL, tex_bvh_node_prim_info, d_bvh.prim_info);
    check_cuda_error("Binding prim_info to texture");

    cudaBindTexture(NULL, tex_tri_xform, d_tri_xform);
    check_cuda_error("Binding tri_xform to texture");

    cudaBindTexture(NULL, tex_tri_normals, d_tri_normals);
    check_cuda_error("Binding tri_normals to texture");

    scoped_timer_stop sts(timers.add("Primary local intersections"));

    assert(d_rays_dir.size() == d_rays_idx.size());

    dvector<unsigned> d_warp_frustum;
    dvector<unsigned> d_ray_warps, d_ray_warp_sizes;

    decompose_into_packets(d_ray_warps, d_ray_warp_sizes, &d_warp_frustum,
                           d_active_ray_packets, d_active_ray_packet_sizes, 32);

//    printf("Warps: %ld\n",d_ray_warps.size());

#if 0
    std::vector<unsigned> h_ray_warp_sizes = to_cpu(d_ray_warp_sizes),
                          h_warp_frustum = to_cpu(d_warp_frustum),
                          h_leaf_sizes = to_cpu(d_active_frustum_leaf_sizes);
    for(int i=0; i<h_ray_warp_sizes.size(); ++i)
    {
        printf("%d -> %d (%d)\n",h_warp_frustum[i], h_ray_warp_sizes[i],
               h_leaf_sizes[h_warp_frustum[i]]);
    }
    exit(0);
#endif

    copy_to_symbol("global_pool_warp_count",d_ray_warps.size());
    copy_to_symbol("global_pool_next_warp",0);


    dim3 dimBlock(32, PERSISTENT_WARPS),
         dimGrid(30);//(d_ray_warps.size()/5 + dimBlock.y-1)/dimBlock.y);

    assert(d_ray_warps.size() == d_ray_warp_sizes.size());
    assert(d_ray_warp_sizes.size() == d_warp_frustum.size());

    dvector<unsigned> d_pos_ray_warp;
    scan_add(d_pos_ray_warp, d_ray_warp_sizes, EXCLUSIVE);

    size_t num_rays = d_pos_ray_warp.back() + d_ray_warp_sizes.back();

    d_isect_positions.resize(num_rays);
    d_isect_normals.resize(num_rays);
    d_isect_rays_idx.resize(num_rays);

    primary_local_intersections<<<dimGrid, dimBlock>>>(
        d_ray_warps, d_ray_warp_sizes,
        d_warp_frustum, 
        d_active_frustum_leaves,
        d_active_frustum_leaf_sizes,
        d_active_idleaves,
        d_rays_idx,
        ray_ori, d_rays_dir,
        d_isect_positions, d_isect_normals, d_isect_rays_idx,
        d_pos_ray_warp);

    check_cuda_error("primary_local_intersections");
//    cudaPrintfDisplay(stdout, false);
//    exit(0);
}

// shadow local intersections
__global__ void shadow_local_intersections(const unsigned *ray_warps,
                                  const unsigned *ray_warp_size,
                                  const unsigned *warp_frustum,
                                  const unsigned *frustum_leaf_base,
                                  const unsigned *frustum_leaf_size,
                                  const unsigned *idleaves,
                                  const float3 *ray_oris,
                                  const float3 *ray_dirs,
                                  const unsigned *ray_idx,
                                  unsigned *ray_on_shadow)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ volatile unsigned next_warp_array[SHADOW_PERSISTENT_WARPS],
                                 warp_count_array[SHADOW_PERSISTENT_WARPS];

    volatile unsigned  &idwarp = next_warp_array[ty],
                       &local_pool_warp_count = warp_count_array[ty];

    if(tx == 0)
        local_pool_warp_count = 0;

    while(true)
    {
        if(local_pool_warp_count == 0 && tx == 0)
        {
            idwarp = atomicAdd(&global_pool_next_warp,SHADOW_BATCH_SIZE);
            local_pool_warp_count = SHADOW_BATCH_SIZE;
        }

        if(idwarp >= global_pool_warp_count)
            return;

        unsigned warp_size = ray_warp_size[idwarp];

        if(tx < warp_size)
        {
            int idray = ray_idx[ray_warps[idwarp] + tx];
            float3 ray_dir = ray_dirs[idray],
                   ray_ori = ray_oris[idray];

            unsigned idfrustum = warp_frustum[idwarp]-1;

            int leaf_count = frustum_leaf_size[idfrustum],
                leaf_base = frustum_leaf_base[idfrustum];

            // for all leaves
            for(int idrefleaf = 0; idrefleaf<leaf_count; ++idrefleaf) 
            {
                int idleaf = idleaves[leaf_base+idrefleaf];

                // ray intersect leaf aabb?
                if(intersects(ray_ori, 1.0f/ray_dir,0,1, 
                              bvh_node_aabb(idleaf)))
                {
                    unsigned prim_info = 
                        tex1Dfetch(tex_bvh_node_prim_info, idleaf);

                    unsigned prim_base = (prim_info>>8)*3,
                             prim_max = prim_base + (prim_info & 0xFF)*3;
                    // for all primitives in this leaf,

                    for(int idprim = prim_base; idprim < prim_max; idprim+=3)
                    {
                        if(intersect(ray_ori,ray_dir,SCENE_EPSILON,1,
                           tex_tri_xform, NULL, idprim))
                        {
                            ray_on_shadow[idray] = true;
                            goto fim_loops;
                        }
                    }
                }
            }
        }

fim_loops:

        if(tx == 0)
        {
            ++idwarp;
            --local_pool_warp_count;
        }
    }
}

__host__ void shadow_local_intersections(
    dvector<unsigned> &d_on_shadow,

    const dvector<unsigned> &d_active_ray_packets,
    const dvector<unsigned> &d_active_ray_packet_sizes,
    const dvector<unsigned> &d_active_frustum_leaves,
    const dvector<unsigned> &d_active_frustum_leaf_sizes,
    const dvector<unsigned> &d_active_idleaves,

    const bvh_soa &d_bvh,

    const dvector<float4> &d_tri_xform,

    const dvector<float3> &d_ray_oris, const dvector<float3> &d_rays_dir,
    const dvector<unsigned> &d_rays_idx)
{
    assert(d_rays_dir.size() == d_rays_idx.size());

    cudaBindTexture(NULL, tex_bvh_node_aabb1, d_bvh.aabb1);
    check_cuda_error("Binding aabb1 to texture");

    cudaBindTexture(NULL, tex_bvh_node_aabb2, d_bvh.aabb2);
    check_cuda_error("Binding aabb2 to texture");

    cudaBindTexture(NULL, tex_bvh_node_prim_info, d_bvh.prim_info);
    check_cuda_error("Binding prim_info to texture");

    cudaBindTexture(NULL, tex_tri_xform, d_tri_xform);
    check_cuda_error("Binding tri_xform to texture");

    dvector<unsigned> d_warp_frustum;
    dvector<unsigned> d_ray_warps, d_ray_warp_sizes;

    decompose_into_packets(d_ray_warps, d_ray_warp_sizes, &d_warp_frustum,
                           d_active_ray_packets, d_active_ray_packet_sizes, 32);

#if 0
    std::vector<unsigned> h_ray_warp_sizes = to_cpu(d_ray_warp_sizes),
                          h_warp_frustum = to_cpu(d_warp_frustum),
                          h_leaf_sizes = to_cpu(d_active_frustum_leaf_sizes);
    for(int i=0; i<h_ray_warp_sizes.size(); ++i)
    {
        printf("%d -> %d (%d)\n",h_warp_frustum[i], h_ray_warp_sizes[i],
               h_leaf_sizes[h_warp_frustum[i]]);
    }
    exit(0);
#endif

    copy_to_symbol("global_pool_warp_count",d_ray_warps.size());
    copy_to_symbol("global_pool_next_warp",0);

    dim3 dimBlock(32, SHADOW_PERSISTENT_WARPS),
         dimGrid(60);//(d_ray_warps.size()/5 + dimBlock.y-1)/dimBlock.y);

    assert(d_ray_warps.size() == d_ray_warp_sizes.size());
    assert(d_ray_warp_sizes.size() == d_warp_frustum.size());

    d_on_shadow.resize(d_rays_dir.size());
    cudaMemset(d_on_shadow, 0, d_on_shadow.size()*sizeof(unsigned));

    shadow_local_intersections<<<dimGrid, dimBlock>>>(
        d_ray_warps, d_ray_warp_sizes,
        d_warp_frustum, 
        d_active_frustum_leaves,
        d_active_frustum_leaf_sizes,
        d_active_idleaves,
        d_ray_oris, d_rays_dir, d_rays_idx,
        d_on_shadow.data());
}

