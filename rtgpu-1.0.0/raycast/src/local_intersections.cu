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

__device__ __constant__ shade::PointLight lights[MAX_LIGHTS];
__device__ __constant__ unsigned num_lights;

inline __device__ AABB bvh_node_aabb(int idnode)
{
    return make_aabb(tex1Dfetch(tex_bvh_node_aabb1, idnode),
                     tex1Dfetch(tex_bvh_node_aabb2, idnode));
}

#define SCENE_EPSILON 0.001

#define PERSISTENT_WARPS 16
#define BATCH_SIZE 2

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
                                  float4 *output)
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

        float3 N = make_float3(0,1,0);
        float t = FLT_MAX;

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
                    if(intersects(ray_ori, 1.0f/ray_dir,SCENE_EPSILON,t, 
                                  bvh_node_aabb(idleaf)))
                    {
                        unsigned prim_info = 
                            tex1Dfetch(tex_bvh_node_prim_info, idleaf);

                        unsigned prim_base = (prim_info>>8);
                        // for all primitives in this leaf,

                        for(int idtri = 0; idtri < (prim_info & 0xFF); ++idtri)
                        {
                            int idprim = (prim_base+idtri)*3;

                            intersect(ray_ori, ray_dir, SCENE_EPSILON, t,
                              tex_tri_xform, tex_tri_normals, idprim,
                              &t, &N);
                        }
                    }
                }
            }

            idrefleaf += bounds_count;
        }

        if(tx < warp_size)
        {
            if(t < FLT_MAX)
            {
                float3 c = make_float3(0.1,0.1,0.1);

                float3 P = ray_ori + t*ray_dir;

                float kd = clamp(dot(N, unit(lights[0].pos-P)), 0.f, 1.f);

                c += make_float3(1)*kd;

                output[idray] = make_float4(c,1);
            }
        }

        if(tx == 0)
        {
            ++idwarp;
            --local_pool_warp_count;
        }
    }
}/*}}}*/

__host__ void primary_local_intersections(
    float4 *d_output,
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

    static dvector<unsigned> d_warp_frustum;
    static dvector<unsigned> d_ray_warps, d_ray_warp_sizes;

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

    primary_local_intersections<<<dimGrid, dimBlock>>>(
        d_ray_warps, d_ray_warp_sizes,
        d_warp_frustum, 
        d_active_frustum_leaves,
        d_active_frustum_leaf_sizes,
        d_active_idleaves,
        d_rays_idx,
        ray_ori, d_rays_dir,
        d_output);
}
#if 0

// shadow local intersections
__global__ void shadow_local_intersections(const unsigned *ray_warps,
                                  const unsigned *ray_warp_size,
                                  const unsigned *warp_frustum,
                                  const unsigned *frustum_leaf_base,
                                  const unsigned *frustum_leaf_size,
                                  const linear_8bvh_leaf *leaves,
                                  const float4 *tri_xform,
                                  const float3 *ray_oris,const float3 *ray_dirs,
                                  unsigned *is_valid_ray)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ volatile unsigned next_warp_array[PERSISTENT_WARPS],
                                 warp_count_array[PERSISTENT_WARPS];

    volatile unsigned  &idwarp = next_warp_array[ty],
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

        int idray = ray_warps[idwarp] + tx; // coalesced
        float3 ray_dir = ray_dirs[idray];
        float3 ray_ori = ray_oris[idray];
        unsigned idfrustum = warp_frustum[idwarp]-1;
        unsigned leaf_count = frustum_leaf_size[idfrustum]; // coalesced
        unsigned first_leaf = frustum_leaf_base[idfrustum]; // coalesced

        int warp_size = ray_warp_size[idwarp];
        bool on_shadow = false;

        // for all leaves
        for(unsigned idleaf = 0; idleaf < leaf_count;)
        {
            __shared__ AABB bounds[PERSISTENT_WARPS*8];
            int bufsize = min(8,leaf_count - idleaf);
            if(tx < bufsize)
                bounds[ty*8+tx] = leaves[first_leaf+idleaf+tx].aabb;

            __threadfence_block();

            if(tx < warp_size)
            {
                for(int i=0; i<bufsize; ++i)
                {
                    int idx = ty*8+i;

                    // ray intersect leaf aabb?
                    if(intersects(ray_ori, 1.0f/ray_dir,SCENE_EPSILON,FLT_MAX, 
                                  bounds[idx]))
                    {
                        unsigned prim_info = leaves[first_leaf+idleaf+i].prim_info;
                        // for all primitives in this leaf,

                        for(unsigned idtri = 0; idtri<(prim_info & 0xFF); ++idtri)
                        {
                            int idx = ((prim_info>>8)+idtri)*3;

                            if(intersect(ray_ori, ray_dir, SCENE_EPSILON, FLT_MAX,
                               &tri_xform[idx]))
                            {
                                on_shadow = true;
                                break;
                            }
                        }
                    }
                }

                if(on_shadow)
                {
                    is_valid_ray[idray] = 1;
                    break;
                }

            }

            idleaf += bufsize;
        }

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
    const dvector<linear_8bvh_leaf> &d_active_leaves,

    const dvector<float4> &d_tri_xform,

    const dvector<float3> &d_ray_oris, const dvector<float3> &d_rays_dir,
    const dvector<unsigned> &d_rays_idx)
{
    assert(d_rays_dir.size() == d_rays_idx.size());

    static dvector<unsigned> d_warp_frustum;
    static dvector<unsigned> d_ray_warps, d_ray_warp_sizes;

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

    printf("Warps: %ld\n",d_ray_warps.size());


    dim3 dimBlock(32, PERSISTENT_WARPS),
         dimGrid(30);//(d_ray_warps.size()/5 + dimBlock.y-1)/dimBlock.y);

    assert(d_ray_warps.size() == d_ray_warp_sizes.size());
    assert(d_ray_warp_sizes.size() == d_warp_frustum.size());

    d_on_shadow.resize(d_rays_dir.size());
    cudaMemset(d_on_shadow, 0, d_on_shadow.size()*sizeof(unsigned));

    shadow_local_intersections<<<dimGrid, dimBlock>>>(
        d_ray_warps, d_ray_warp_sizes,
        d_warp_frustum, 
        d_active_frustum_leaves,
        d_active_frustum_leaf_sizes,
        d_active_leaves,
        d_tri_xform,
        d_ray_oris, d_rays_dir,
        d_on_shadow.data());
}

#endif
