#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <math_constants.h>
#include "ray_compression.h"
#include "packet_decomposition.h"
#include "local_intersections.h"
#include "frustum.h"
#include "dvector.h"
#include "util.h"
#include "float.h"
#include "raytrace.h"
#include "sphere.cu"
#include "light.h"
#include "bvh.h"
#include "triangle.cu"
#include "timer.h"
#include "traversal.h"
#include "cuda_bvh.h"
#include "cuPrintf.cu"
#include <curand_kernel.h>

__device__ __constant__ shade::PointLight lights[MAX_LIGHTS];
__device__ __constant__ unsigned num_lights;

#define PACKET_SIZE 256

#define UCF 0.004

#define TRACE_FRUSTUM_SIZE 0

#define SHADOW_STRATA 16
#define SQRT_SHADOW_STRATA 4
#define SHADOW_SAMPLES_PER_STRATUM 1
#define SHADOW_TOTAL_SAMPLES (SHADOW_STRATA*SHADOW_SAMPLES_PER_STRATUM)

__device__ unsigned spread2(unsigned x)/*{{{*/
{
                                     // ................FEDCBA9876543210
    x = (x | (x << 8)) & 0x00FF00FF; // ........FEDCBA98........76543210
    x = (x | (x << 4)) & 0x0F0F0F0F; // ....FEDC....BA98....7654....3210
    x = (x | (x << 2)) & 0x33333333; // ..FE..DC..BA..98..76..54..32..10
    x = (x | (x << 1)) & 0x55555555; // .F.E.D.C.B.A.9.8.7.6.5.4.3.2.1.0
    return x;
}/*}}}*/
__device__ unsigned zorder_hash(unsigned x, unsigned y)/*{{{*/
{
    return spread2(x) | (spread2(y)<<1);
}/*}}}*/
__device__ unsigned spread3(unsigned x)/*{{{*/
{
                                      // ......................9876543210
    x = (x | (x << 10)) & 0x000F801F; // ............98765..........43210
    x = (x | (x <<  4)) & 0x00E181C3; // ........987....56......432....10    
    x = (x | (x <<  2)) & 0x03248649; // ......98..7..5..6....43..2..1..0
    x = (x | (x <<  2)) & 0x09249249; // ....9..8..7..5..6..4..3..2..1..0
    return x;
}/*}}}*/
__device__ unsigned zorder_hash(unsigned x, unsigned y, unsigned z)/*{{{*/
{
    return spread3(x) | (spread3(y)<<1) | (spread3(z)<<2);
}/*}}}*/

//{{{ create primary rays -------------------------------------------------
__global__ void create_primary_rays(float3 U, float3 V, float3 W,
                                    int width, int height, 
                                    float inv_cell_size,
                                    float3 *rays_dir,
                                    unsigned *rays_hash)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x,  by = blockIdx.y;

    int row = (by<<4) + ty,
        col = (bx<<4) + tx;

    if(row >= height || col >= width)
        return;

    // [0,width],[0,height] -> [-1,1],[-1,1]
    float2 d = make_float2(col,row) / make_float2(width,height)*2 - 1;

    int idx = row*width + col;
    rays_dir[idx] = unit(d.x*U + d.y*V + W);

    float theta = atan(d.x), // atan2f(d.x,1.0f)
          phi = atan(d.y*rsqrt(d.x*d.x+1.0f));

    uint2 cell;
    cell.x = (phi+CUDART_PI_F)/(2*CUDART_PI_F)*inv_cell_size;
    cell.y = (theta+CUDART_PI_F)/(2*CUDART_PI_F)*inv_cell_size;

    rays_hash[idx] = zorder_hash(cell.x,cell.y);
}

__host__ float create_primary_rays(dvector<float3> &d_rays_dir,
                                  dvector<unsigned> &d_rays_hash,
                                  const AABB &scene_bounds,
                                  float3 U, float3 V, float3 W,
                                  int width, int height)
{
    dim3 dimBlock(16,16);
    dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x,
                 (height + dimBlock.y-1)/dimBlock.y);

    d_rays_dir.resize(width*height);
    d_rays_hash.resize(width*height);

    float cell_size = length(scene_bounds.hsize)*2 * UCF;

    scoped_timer_stop sts(timers.add("Primary ray creation"));

    create_primary_rays<<<dimGrid,dimBlock>>>(U,V,W,width,height, 
                                              1.0f/cell_size,
                                      d_rays_dir, d_rays_hash);

    return cell_size;
}
/*}}}*/

//{{{ create shadow rays -------------------------------------------------
__global__ void create_shadow_rays(const float3 *positions, 
                                   unsigned count,
                                   float3 scene_lower,
                                   float inv_cell_size,
                                   float3 *rays_dir,
                                   unsigned *rays_hash)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    float3 ray_ori = positions[idx];

    rays_dir[idx] = lights[0].pos - ray_ori;

    uint3 pcell = make_uint3((ray_ori - scene_lower)*inv_cell_size);

    rays_hash[idx] = zorder_hash(pcell.x, pcell.y, pcell.z);

}

__host__ float create_shadow_rays(dvector<float3> &d_rays_dir,/*{{{*/
                                 dvector<unsigned> &d_rays_hash,
                                 const AABB &scene_bounds,
                                 const dvector<float3> &positions)
{
    dim3 dimBlock(256);
    dim3 dimGrid((positions.size()+255)/256);

    d_rays_dir.resize(positions.size());
    d_rays_hash.resize(positions.size());

    float cell_size = length(scene_bounds.hsize)*2 * UCF;

//    std::cout  << "Cell size: " << cell_size << std::endl;

    create_shadow_rays<<<dimGrid,dimBlock>>>(
        positions, positions.size(),
        scene_bounds.center-scene_bounds.hsize, 1/cell_size,
        d_rays_dir, d_rays_hash);

    return cell_size;
}/*}}}*/
/*}}}*/

//{{{ create soft_shadow rays -------------------------------------------------
__global__ void create_soft_shadow_rays(const float3 *positions, 
                                   unsigned count,
                                   float3 scene_lower,
                                   float inv_cell_size,
                                   float3 *rays_ori,
                                   float3 *rays_dir,
                                   unsigned *rays_hash)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    float3 ray_ori = positions[idx];

    uint3 pcell = make_uint3((ray_ori - scene_lower)*inv_cell_size);


    unsigned hash = zorder_hash(pcell.x, pcell.y, pcell.z);

//    cuPrintf("pcell: %d,%d,%d - %x, %d\n",pcell.x, pcell.y, pcell.z, hash, (unsigned int)floor(log2((float)hash))+1);

    curandStateXORWOW_t state;
    curand_init(idx, 0, 0, &state);

    int idxray = idx*SHADOW_TOTAL_SAMPLES;

    Sphere s;
    s.center = lights[0].pos;
    s.radius = 0.005;

    int idstratum=0;

    for(int sx=0; sx<SQRT_SHADOW_STRATA; ++sx)
    {
        for(int sy=0; sy<SQRT_SHADOW_STRATA; ++sy, ++idstratum)
        {
            for(int i=0; i<SHADOW_SAMPLES_PER_STRATUM; ++i)
            {
                float u1 = (curand_uniform(&state)+sx)/SQRT_SHADOW_STRATA,
                      u2 = (curand_uniform(&state)+sy)/SQRT_SHADOW_STRATA;

                float3 ps = sample_surface(s, ray_ori, u1, u2);

                float3 dir = ps-ray_ori;

                rays_dir[idxray] = dir;
                rays_ori[idxray] = ray_ori;

                dir = unit(dir);

                float theta = atan(dir.x), // atan2f(d.x,1.0f)
                      phi = atan(dir.y*rsqrt(dir.x*dir.x+1.0f));

                uint2 cell;
                cell.x = (phi+CUDART_PI_F)/(2*CUDART_PI_F)*8;
                cell.y = (theta+CUDART_PI_F)/(2*CUDART_PI_F)*8;

                rays_hash[idxray] = hash | (zorder_hash(cell.x,cell.y) << 24);

//                rays_hash[idxray] = hash | ((idstratum>>3) << 24);

                idxray++;
            }
        }
    }
}

__host__ float create_soft_shadow_rays(dvector<float3> &d_rays_ori,
                                       dvector<float3> &d_rays_dir,
                                 dvector<unsigned> &d_rays_hash,
                                 const AABB &scene_bounds,
                                 const dvector<float3> &positions)
{
    dim3 dimBlock(256);
    dim3 dimGrid((positions.size()+255)/256);

    d_rays_dir.resize(positions.size()*SHADOW_TOTAL_SAMPLES);
    d_rays_hash.resize(positions.size()*SHADOW_TOTAL_SAMPLES);
    d_rays_ori.resize(positions.size()*SHADOW_TOTAL_SAMPLES);

    float cell_size = length(scene_bounds.hsize)*2 * UCF;

//    std::cout << "cell_size: " << cell_size << std::endl;
//    std::cout << "# cells: " << scene_bounds.hsize*2/cell_size << std::endl;

    create_soft_shadow_rays<<<dimGrid,dimBlock>>>(
        positions, positions.size(),
        scene_bounds.center-scene_bounds.hsize, 1/cell_size,
        d_rays_ori, d_rays_dir, d_rays_hash);

//    cudaPrintfDisplay(stdout, false);
//    exit(0);

    return cell_size;
}
/*}}}*/

//{{{ decompress rays ------------------------------------------------------
__global__ void decompress_rays(const unsigned *chunk_idx,
                                const unsigned *chunk_base,
                                const unsigned *scan_sorted_chunk_size,
                                size_t count,
                                unsigned *skeleton, 
                                unsigned *head_flags,
                                size_t count_skel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    int idx_skel = scan_sorted_chunk_size[idx];
    int idx_sorted = chunk_idx[idx];

    skeleton[idx_skel] = chunk_base[idx_sorted];
    head_flags[idx_skel] = 1;
}

__host__ void decompress_rays(dvector<unsigned> &rays_idx,
                              const dvector<unsigned> &comp_hash, 
                              const dvector<unsigned> &comp_base, 
                              const dvector<unsigned> &comp_size, 
                              const dvector<unsigned> &comp_idx,
                              size_t ray_count)
{
    assert(comp_hash.size() == comp_base.size());
    assert(comp_base.size() == comp_size.size());
    assert(comp_size.size() == comp_idx.size());

    dvector<unsigned> scan_comp_size;
    scan_add(scan_comp_size, comp_size, EXCLUSIVE);

    dvector<unsigned> skeleton, head_flags;
    init_skeleton(skeleton, head_flags, 1, ray_count);

    dim3 dimGrid, dimBlock;
    compute_linear_grid(comp_hash.size(), dimGrid, dimBlock);

    decompress_rays<<<dimGrid, dimBlock>>>(
        comp_idx, comp_base, scan_comp_size, comp_hash.size(),
        skeleton, head_flags, ray_count); 

    segscan_add(rays_idx, skeleton, head_flags);
}/*}}}*/

//{{{ sort_rays --------------------------------------------------------------
__global__ void calc_sorted_chunk_size(const unsigned *chunk_idx,
                                       const unsigned *chunk_base,
                                       unsigned *chunk_size,
                                       size_t count, size_t input_item_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    int orig_idx = chunk_idx[idx];

    if(orig_idx == count-1)
        chunk_size[idx] = input_item_count - chunk_base[orig_idx];
    else
        chunk_size[idx] = chunk_base[orig_idx+1] - chunk_base[orig_idx];
}

__host__ void sort_rays(dvector<unsigned> &hashes,
                        dvector<unsigned> &bases, // bases aren't sorted
                        dvector<unsigned> &sizes,
                        dvector<unsigned> &indices,
                        size_t ray_count, int bits)
{
    assert(hashes.size() == bases.size());
    assert(bases.size() == indices.size());

    sort(hashes, indices, bits);

    sizes.resize(hashes.size());

    dim3 dimGrid, dimBlock;
    compute_linear_grid(hashes.size(), dimGrid, dimBlock);

    calc_sorted_chunk_size<<<dimGrid, dimBlock>>>(
        indices, bases, sizes, indices.size(), ray_count);

}/*}}}*/

//{{{ create primary frustums ------------------------------------------------
__host__ __device__ int dominant_axis(float3 v)/*{{{*/
{
    float x = fabs(v.x),
          y = fabs(v.y),
          z = fabs(v.z);

    if(x > y && x > z)
        return v.x >= 0 ? 1 : -1;
    else if(y > z)
        return v.y >= 0 ? 2 : -2;
    else
        return v.z >= 0 ? 3 : -3;
}/*}}}*/
__host__ __device__ void min_max(float &umin, float &umax, float u)/*{{{*/
{
    if(u < umin)
        umin = u;

    if(u > umax)
        umax = u;
}/*}}}*/

__device__ void reduce_min(int idray, int tx, float &vmin, float buffer[256])/*{{{*/
{
    buffer[tx] = vmin;

    if(idray < 16) 
        buffer[tx] = min(buffer[tx], buffer[tx+16]);

    if(idray < 8) 
        buffer[tx] = min(buffer[tx], buffer[tx+8]);

    if(idray < 4) 
        buffer[tx] = min(buffer[tx], buffer[tx+4]);

    if(idray < 2) 
        buffer[tx] = min(buffer[tx], buffer[tx+2]);

    if(idray < 1) 
        buffer[tx] = min(buffer[tx], buffer[tx+1]);

    vmin = buffer[tx];
}/*}}}*/
__device__ void reduce_max(int idray, int tx, float &vmax, float buffer[256])/*{{{*/
{
    buffer[tx] = vmax;

    if(idray < 16) 
        buffer[tx] = max(buffer[tx], buffer[tx+16]);

    if(idray < 8) 
        buffer[tx] = max(buffer[tx], buffer[tx+8]);

    if(idray < 4) 
        buffer[tx] = max(buffer[tx], buffer[tx+4]);

    if(idray < 2) 
        buffer[tx] = max(buffer[tx], buffer[tx+2]);

    if(idray < 1) 
        buffer[tx] = max(buffer[tx], buffer[tx+1]);

    vmax = buffer[tx];
}/*}}}*/

__device__ float4 create_plane(float3 ray_ori, int s, float a, float b, float c)/*{{{*/
{
    float3 n = unit(s >= 0 ? make_float3(a,b,c) : -make_float3(a,b,c));
    float d = -dot(ray_ori, n);
    return make_float4(n,d);
}/*}}}*/

__device__ unsigned calc_dirsign(int s, float x, float y, float z)/*{{{*/
{
    unsigned bits=0;
    if(x < 0)
        bits |= 1;
    if(y < 0)
        bits |= 2;
    if(z < 0)
        bits |= 4;

    if(s < 0)
        bits = ~bits & 7;

    return bits;
}/*}}}*/

__global__ void create_frustums(FrustumsGPU frustums,/*{{{*/
                                const float3 ray_ori,
                                const unsigned *rays_idx,
                                const float3 *rays_dir,
                                const unsigned *packet_indices,
                                const unsigned *packet_sizes,
                                unsigned num_packets)
{
    int tx = threadIdx.x, bx = blockIdx.x;

    int idfrustum = (bx<<3) + (tx>>5);

    if(idfrustum >= num_packets)
        return;

    // calculo do eixo dominante
    // assumimos que todos os raios tem o mesmo eixo dominante, já que são
    // coerentes

    unsigned packet_size = packet_sizes[idfrustum],
             idfirst_ray = packet_indices[idfrustum];

    int z_axis = dominant_axis(rays_dir[rays_idx[idfirst_ray]]);

    // calcula u_min/max e v_min/max 


    // 32 threads vao processar um frustum
    // cada frustum tem no máximo 256 raios, portanto cada thread vai processar
    // no máximo 8 raios.

    float u_min, v_min, u_max, v_max;
    u_min = v_min = FLT_MAX;
    u_max = v_max = -FLT_MAX;

    int idray = tx&31;

    switch(abs(z_axis))
    {
    case 1: // x-axis is dominant
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            float3 dir = rays_dir[rays_idx[idfirst_ray + idray + i]];

            float inv = 1.0f/dir.x;

            min_max(u_min, u_max, dir.y * inv);
            min_max(v_min, v_max, dir.z * inv);
        }
        break;
    case 2: // y-axis is dominant
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            float3 dir = rays_dir[rays_idx[idfirst_ray + idray+i]];

            float inv = 1.0f/dir.y;

            min_max(u_min, u_max, dir.x * inv);
            min_max(v_min, v_max, dir.z * inv);
        }
        break;
    case 3: // z-axis is dominant
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            float3 dir = rays_dir[rays_idx[idfirst_ray + idray +i]];

            float inv = 1.0f/dir.z;

            min_max(u_min, u_max, dir.x * inv);
            min_max(v_min, v_max, dir.y * inv);
        }
        break;
    }

    __shared__ float buffer[256];

    reduce_min(idray, tx, u_min, buffer);
    reduce_max(idray, tx, u_max, buffer);

    reduce_min(idray, tx, v_min, buffer);
    reduce_max(idray, tx, v_max, buffer);

    if(idray != 0)
        return;

    if(abs(u_max-u_min) < 1e-5f)
    {
        u_max = u_min+1e-5f;
        u_min -= 1e-5f;
    }
    if(abs(v_max - v_min) < 1e-5f)
    {
        v_max = v_min+1e-5f;
        v_min -= 1e-5f;
    }

    float3 O = ray_ori;

    switch(abs(z_axis))
    {
    case 1: // X is major axis
        frustums.dirsign[idfrustum] = calc_dirsign(z_axis, 1,u_min+u_max, v_min+v_max);
        // top (towards +Z)
        frustums.top[idfrustum]    = create_plane(O, z_axis,-v_max,  0,  1);
        // right (towards -Y)
        frustums.right[idfrustum]  = create_plane(O, z_axis, u_min, -1,  0);
        // bottom (towards -Z)
        frustums.bottom[idfrustum] = create_plane(O, z_axis, v_min,  0, -1);
        // left (towards +Y)
        frustums.left[idfrustum]   = create_plane(O, z_axis,-u_max,  1,  0);

        break;
    case 2: // Y is major axis
        frustums.dirsign[idfrustum] = calc_dirsign(z_axis, u_min+u_max, 1, v_min+v_max);

        // top (towards +Z)
        frustums.top[idfrustum]    = create_plane(O, z_axis, 0, -v_max,  1);
        // right (towards +X)
        frustums.right[idfrustum]  = create_plane(O, z_axis, 1, -u_max,  0);
        // bottom (towards -Z)
        frustums.bottom[idfrustum] = create_plane(O, z_axis, 0,  v_min, -1);
        // left (towards -X)
        frustums.left[idfrustum]   = create_plane(O, z_axis,-1,  u_min,  0);

        break;
    case 3: // Z is major axis
        frustums.dirsign[idfrustum] = calc_dirsign(z_axis, u_min+u_max, v_min+v_max, 1);

        // top (towards +Y)
        frustums.top[idfrustum]    = create_plane(O, z_axis, 0,  1, -v_max);
        // right (towards -X)
        frustums.right[idfrustum]  = create_plane(O, z_axis,-1,  0,  u_min);
        // bottom (towards -Y)
        frustums.bottom[idfrustum] = create_plane(O, z_axis, 0, -1,  v_min);
        // left (towards +X)
        frustums.left[idfrustum]   = create_plane(O, z_axis, 1,  0, -u_max);
        break;
    }
}/*}}}*/
__host__ void create_frustums(Frustums &frustums,/*{{{*/
                              const float3 &ray_ori,
                              const dvector<unsigned> &rays_idx,
                              const dvector<float3> &rays_dir,
                              const dvector<unsigned> &packet_indices,
                              const dvector<unsigned> &packet_sizes)
{
    frustums.resize(packet_indices.size());

    dim3 dimBlock(256),
         dimGrid((packet_indices.size()+7)/8);

    FrustumsGPU fgpu = frustums;

    scoped_timer_stop sts(timers.add("Primary frustums creation"));

    create_frustums<<<dimGrid, dimBlock>>>(fgpu, ray_ori, rays_idx, rays_dir, 
                                           packet_indices,
                                           packet_sizes, packet_indices.size());
}/*}}}*/
//}}}

//{{{ create secondary frustums -------------------------------------------

__device__ float4 create_plane(float3 ray_ori, float a, float b, float c)/*{{{*/
{
    float3 n = unit(make_float3(a,b,c));
    float d = -dot(ray_ori, n);
    return make_float4(n,d);
}/*}}}*/
__device__ unsigned calc_dirsign(float x, float y, float z)/*{{{*/
{
    unsigned bits=0;
    if(x < 0)
        bits |= 1;
    if(y < 0)
        bits |= 2;
    if(z < 0)
        bits |= 4;

    return bits;
}/*}}}*/

__global__ void create_frustums(FrustumsOriGPU frustums,/*{{{*/
                                const AABB scene_bounds,
                                const unsigned *rays_idx,
                                const float3 *rays_ori,
                                const float3 *rays_dir,
                                const unsigned *packet_indices,
                                const unsigned *packet_sizes,
                                unsigned num_packets)
{
    // 32 threads vao processar um frustum, um bloco processa 8 frustums
    // cada frustum tem no máximo 256 raios, portanto cada thread vai processar
    // no máximo 8 raios.

    int tx = threadIdx.x, bx = blockIdx.y*gridDim.x + blockIdx.x;

    int idfrustum = (bx<<3) + (tx>>5);

    if(idfrustum >= num_packets)
        return;

    __shared__ float buffer[256];

    int idray = tx&31;

    // calculo do eixo dominante
    // assumimos que todos os raios tem o mesmo eixo dominante, já que são
    // coerentes

    unsigned packet_size = packet_sizes[idfrustum],
             idfirst_ray = packet_indices[idfrustum];

    int z_axis = dominant_axis(rays_dir[rays_idx[idfirst_ray]]);

    // posicao do plano near (bounds das origens na direcao -z_axis)

    float near_plane, far_frustum;
    switch(z_axis)
    {
    case 1: // x-axis is dominant
        near_plane = FLT_MAX;
        far_frustum = -FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];
            float ori = rays_ori[idrayabs].x;
            near_plane = min(near_plane, ori);
            far_frustum = max(far_frustum, ori+rays_dir[idrayabs].x);
        }
        reduce_min(idray, tx, near_plane, buffer);
        reduce_max(idray, tx, far_frustum, buffer);
        break;
    case -1: // -x-axis is dominant
        near_plane = -FLT_MAX;
        far_frustum = FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float ori = rays_ori[idrayabs].x;
            near_plane = max(near_plane, ori);
            far_frustum = min(far_frustum, ori+rays_dir[idrayabs].x);
        }
        reduce_max(idray, tx, near_plane, buffer);
        reduce_min(idray, tx, far_frustum, buffer);
        break;
    case 2: // y-axis is dominant
        near_plane = FLT_MAX;
        far_frustum = -FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float ori = rays_ori[idrayabs].y;
            near_plane = min(near_plane, ori);
            far_frustum = max(far_frustum, ori+rays_dir[idrayabs].y);
        }
        reduce_min(idray, tx, near_plane, buffer);
        reduce_max(idray, tx, far_frustum, buffer);
        break;
    case -2: // -y-axis is dominant
        near_plane = -FLT_MAX;
        far_frustum = FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float ori = rays_ori[idrayabs].y;
            near_plane = max(near_plane, ori);
            far_frustum = min(far_frustum, ori+rays_dir[idrayabs].y);
        }
        reduce_max(idray, tx, near_plane, buffer);
        reduce_min(idray, tx, far_frustum, buffer);
        break;
    case 3: // z-axis is dominant
        near_plane = FLT_MAX;
        far_frustum = -FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float ori = rays_ori[idrayabs].z;
            near_plane = min(near_plane, ori);
            far_frustum = max(far_frustum, ori+rays_dir[idrayabs].z);
        }
        reduce_min(idray, tx, near_plane, buffer);
        reduce_max(idray, tx, far_frustum, buffer);
        break;
    case -3: // -z-axis is dominant
        near_plane = -FLT_MAX;
        far_frustum = FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float ori = rays_ori[idrayabs].z;
            near_plane = max(near_plane, ori);
            far_frustum = min(far_frustum, ori+rays_dir[idrayabs].z);
        }
        reduce_max(idray, tx, near_plane, buffer);
        reduce_min(idray, tx, far_frustum, buffer);
        break;
    }

    float far_u_min, far_v_min, far_u_max, far_v_max;
    far_u_min = far_v_min = FLT_MAX;
    far_u_max = far_v_max = -FLT_MAX;

    float near_u_min, near_v_min, near_u_max, near_v_max;
    near_u_min = near_v_min = FLT_MAX;
    near_u_max = near_v_max = -FLT_MAX;

    float far_plane;
    switch(abs(z_axis))
    {
    case 1: // x-axis is dominant
        far_plane = scene_bounds.center.x + copysignf(scene_bounds.hsize.x,z_axis);
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float3 ori = rays_ori[idrayabs];

            float3 dir = rays_dir[idrayabs];
            float inv = 1.0f/dir.x;

            float fp = (far_plane-ori.x)*inv;
            min_max(far_u_min, far_u_max, ori.y + dir.y*fp);
            min_max(far_v_min, far_v_max, ori.z + dir.z*fp);

            float np = (near_plane-ori.x)*inv;
            min_max(near_u_min, near_u_max, ori.y + dir.y*np);
            min_max(near_v_min, near_v_max, ori.z + dir.z*np);

        }
        break;
    case 2: // y-axis is dominant
        far_plane = scene_bounds.center.y + copysignf(scene_bounds.hsize.y,z_axis);
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float3 ori = rays_ori[idrayabs];

            float3 dir = rays_dir[idrayabs];
            float inv = 1.0f/dir.y;

            float fp = (far_plane-ori.y)*inv;
            min_max(far_u_min, far_u_max, ori.x + dir.x*fp);
            min_max(far_v_min, far_v_max, ori.z + dir.z*fp);

            float np = (near_plane-ori.y)*inv;
            min_max(near_u_min, near_u_max, ori.x + dir.x*np);
            min_max(near_v_min, near_v_max, ori.z + dir.z*np);
        }
        break;
    case 3: // z-axis is dominant
        far_plane = scene_bounds.center.z + copysignf(scene_bounds.hsize.z,z_axis);
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            int idrayabs = rays_idx[idfirst_ray + idray + i];

            float3 ori = rays_ori[idrayabs];

            float3 dir = rays_dir[idrayabs];
            float inv = 1.0f/dir.z;

            float fp = (far_plane-ori.z)*inv;
            min_max(far_u_min, far_u_max, ori.x + dir.x*fp);
            min_max(far_v_min, far_v_max, ori.y + dir.y*fp);

            float np = (near_plane-ori.z)*inv;
            min_max(near_u_min, near_u_max, ori.x + dir.x*np);
            min_max(near_v_min, near_v_max, ori.y + dir.y*np);
        }
        break;
    }

    reduce_min(idray, tx, far_u_min, buffer);
    reduce_max(idray, tx, far_u_max, buffer);

    reduce_min(idray, tx, far_v_min, buffer);
    reduce_max(idray, tx, far_v_max, buffer);

    reduce_min(idray, tx, near_u_min, buffer);
    reduce_max(idray, tx, near_u_max, buffer);

    reduce_min(idray, tx, near_v_min, buffer);
    reduce_max(idray, tx, near_v_max, buffer);

    if(idray != 0)
        return;

    if(abs(far_u_max - far_u_min) < 1e-5f)
    {
        far_u_max = far_u_min+1e-5f;
        far_u_min -= 1e-5f;
    }
    if(abs(far_v_max - far_v_min) < 1e-5f)
    {
        far_v_max = far_v_min+1e-5f;
        far_v_min -= 1e-5f;
    }

    float3 O;
    float d = far_plane - near_plane;
    
    switch(z_axis)
    {
    case 1: // X is major axis, right is -Y, top is +Z
        // top
        O = make_float3(near_plane, near_u_min, near_v_max);
        frustums.top[idfrustum]   
            = create_plane(O, -(far_v_max-near_v_max), 0,d);
        // right
        frustums.right[idfrustum] 
            = create_plane(O, (far_u_min-near_u_min),-d,0);

        // bottom
        O = make_float3(near_plane, near_u_max, near_v_min);
        frustums.bottom[idfrustum]
            = create_plane(O, (far_v_min-near_v_min),0,-d);

        // left
        frustums.left[idfrustum]  
            = create_plane(O, -(far_u_max-near_u_max), d,0);

        // near
        frustums.near[idfrustum]
            = create_plane(make_float3(near_plane,0,0),-1,0,0);

        // far
        frustums.far[idfrustum]
            = create_plane(make_float3(far_frustum,0,0),1,0,0);

#if TRACE_FRUSTUM_SIZE
        float ang = fabs(dot(make_float3(frustums.top[idfrustum]),
                        make_float3(frustums.bottom[idfrustum]))) +
                    fabs(dot(make_float3(frustums.left[idfrustum]),
                        make_float3(frustums.right[idfrustum])));
        ang /= 2;
        cuPrintf("%f\n",acos(min(max(ang,-1.0f),1.0f))/M_PI*180);
#endif

        frustums.dirsign[idfrustum] 
            = calc_dirsign(d, (far_u_max+far_u_min)-(near_u_max+near_u_min),
                              (far_v_max+far_v_min)-(near_v_max+near_v_min));
        break;
    case -1: // -X is major axis, right is +Y, top is +Z
        // top
        O = make_float3(near_plane, near_u_max, near_v_max);
        frustums.top[idfrustum]   
            = create_plane(O, (far_v_max-near_v_max), 0,-d);
        // right
        frustums.right[idfrustum] 
            = create_plane(O, (far_u_max-near_u_max),-d,0);

        // bottom
        O = make_float3(near_plane, near_u_min, near_v_min);
        frustums.bottom[idfrustum]
            = create_plane(O, -(far_v_min-near_v_min),0,d);

        // left
        frustums.left[idfrustum]  
            = create_plane(O, -(far_u_min-near_u_min), d,0);

        // near
        frustums.near[idfrustum]
            = create_plane(make_float3(near_plane,0,0),1,0,0);

        // far
        frustums.far[idfrustum]
            = create_plane(make_float3(far_frustum,0,0),-1,0,0);

        frustums.dirsign[idfrustum] 
            = calc_dirsign(d, (far_u_max+far_u_min)-(near_u_max+near_u_min),
                              (far_v_max+far_v_min)-(near_v_max+near_v_min));

#if TRACE_FRUSTUM_SIZE
        ang = fabs(dot(make_float3(frustums.top[idfrustum]),
                       make_float3(frustums.bottom[idfrustum]))) +
              fabs(dot(make_float3(frustums.left[idfrustum]),
                       make_float3(frustums.right[idfrustum])));
        ang /= 2;
        cuPrintf("%f\n",acos(min(max(ang,-1.0f),1.0f))/M_PI*180);
#endif

        break;
    case 2: // Y is major axis, right is +X, top is +Z
        // top
        O = make_float3(near_u_max, near_plane, near_v_max);
        frustums.top[idfrustum]   
            = create_plane(O, 0, -(far_v_max-near_v_max), d);

        // right
        frustums.right[idfrustum] 
            = create_plane(O, d, -(far_u_max-near_u_max), 0);

        // bottom
        O = make_float3(near_u_min, near_plane, near_v_min);
        frustums.bottom[idfrustum]
            = create_plane(O, 0, (far_v_min-near_v_min),-d);

        // left
        frustums.left[idfrustum]  
            = create_plane(O, -d, (far_u_min-near_u_min), 0);

        // near
        frustums.near[idfrustum]
            = create_plane(make_float3(0,near_plane,0),0,-1,0);

        // far
        frustums.far[idfrustum]
            = create_plane(make_float3(0,far_frustum,0),0,1,0);

        frustums.dirsign[idfrustum] 
            = calc_dirsign((far_u_max+far_u_min)-(near_u_max+near_u_min),
                           d,
                           (far_v_max+far_v_min)-(near_v_max+near_v_min));

#if TRACE_FRUSTUM_SIZE
        ang = fabs(dot(make_float3(frustums.top[idfrustum]),
                       make_float3(frustums.bottom[idfrustum]))) +
              fabs(dot(make_float3(frustums.left[idfrustum]),
                       make_float3(frustums.right[idfrustum])));
        ang /= 2;
        cuPrintf("%f\n",acos(min(max(ang,-1.0f),1.0f))/M_PI*180);
#endif

        break;
    case -2: // -Y is major axis, right is -X, top is +Z
        // top
        O = make_float3(near_u_min, near_plane, near_v_max);
        frustums.top[idfrustum]   
            = create_plane(O, 0, (far_v_max-near_v_max), -d);

        // right
        frustums.right[idfrustum] 
            = create_plane(O, d, -(far_u_min-near_u_min), 0);

        // bottom
        O = make_float3(near_u_max, near_plane, near_v_min);
        frustums.bottom[idfrustum]
            = create_plane(O, 0, -(far_v_min-near_v_min), d);

        // left
        frustums.left[idfrustum]  
            = create_plane(O, -d, (far_u_max-near_u_max), 0);

        // near
        frustums.near[idfrustum]
            = create_plane(make_float3(0,near_plane,0),0,1,0);

        // far
        frustums.far[idfrustum]
            = create_plane(make_float3(0,far_frustum,0),0,-1,0);

        frustums.dirsign[idfrustum] 
            = calc_dirsign((far_u_max+far_u_min)-(near_u_max+near_u_min),
                           d,
                           (far_v_max+far_v_min)-(near_v_max+near_v_min));
#if TRACE_FRUSTUM_SIZE
        ang = fabs(dot(make_float3(frustums.top[idfrustum]),
                       make_float3(frustums.bottom[idfrustum]))) +
              fabs(dot(make_float3(frustums.left[idfrustum]),
                       make_float3(frustums.right[idfrustum])));
        ang /= 2;
        cuPrintf("%f\n",acos(min(max(ang,-1.0f),1.0f))/M_PI*180);
#endif

        break;
    case 3: // Z is major axis, right is -X, top is +Y
        // top
        O = make_float3(near_u_min, near_v_max, near_plane);
        frustums.top[idfrustum]   
            = create_plane(O, 0, d, -(far_v_max-near_v_max));

        // right
        frustums.right[idfrustum] 
            = create_plane(O, -d, 0, (far_u_min-near_u_min));

        // bottom
        O = make_float3(near_u_max, near_v_min, near_plane);
        frustums.bottom[idfrustum]
            = create_plane(O, 0,-d, (far_v_min-near_v_min));

        // left
        frustums.left[idfrustum]  
            = create_plane(O, d, 0, -(far_u_max-near_u_max));

        // near
        frustums.near[idfrustum]
            = create_plane(make_float3(0,0,near_plane),0,0,-1);

        // far
        frustums.far[idfrustum]
            = create_plane(make_float3(0,0,far_frustum),0,0,1);

        frustums.dirsign[idfrustum] 
            = calc_dirsign((far_u_max+far_u_min)-(near_u_max+near_u_min),
                           (far_v_max+far_v_min)-(near_v_max+near_v_min),
                           d);
#if TRACE_FRUSTUM_SIZE
        ang = fabs(dot(make_float3(frustums.top[idfrustum]),
                       make_float3(frustums.bottom[idfrustum]))) +
              fabs(dot(make_float3(frustums.left[idfrustum]),
                       make_float3(frustums.right[idfrustum])));
        ang /= 2;
        cuPrintf("%f\n",acos(min(max(ang,-1.0f),1.0f))/M_PI*180);
#endif

        break;
    case -3: // -Z is major axis, right is +X, top is +Y
        // top
        O = make_float3(near_u_max, near_v_max, near_plane);
        frustums.top[idfrustum]   
            = create_plane(O, 0, -d, (far_v_max-near_v_max));

        // right
        frustums.right[idfrustum] 
            = create_plane(O, -d, 0, (far_u_max-near_u_max));

        // bottom
        O = make_float3(near_u_min, near_v_min, near_plane);
        frustums.bottom[idfrustum]
            = create_plane(O, 0,d, -(far_v_min-near_v_min));

        // left
        frustums.left[idfrustum]  
            = create_plane(O, d, 0, -(far_u_min-near_u_min));

        // near
        frustums.near[idfrustum]
            = create_plane(make_float3(0,0,near_plane),0,0,1);

        // far
        frustums.far[idfrustum]
            = create_plane(make_float3(0,0,far_frustum),0,0,-1);

        frustums.dirsign[idfrustum] 
            = calc_dirsign((far_u_max+far_u_min)-(near_u_max+near_u_min),
                           (far_v_max+far_v_min)-(near_v_max+near_v_min),
                           d);
#if TRACE_FRUSTUM_SIZE
        ang = fabs(dot(make_float3(frustums.top[idfrustum]),
                       make_float3(frustums.bottom[idfrustum]))) +
              fabs(dot(make_float3(frustums.left[idfrustum]),
                       make_float3(frustums.right[idfrustum])));
        ang /= 2;
        cuPrintf("%f\n",acos(min(max(ang,-1.0f),1.0f))/M_PI*180);
#endif

        break;
    }

#if 0
    cuPrintf("%i: %f,%f,%f - %f\n",idfrustum,
             frustums.far[idfrustum].x,
             frustums.far[idfrustum].y,
             frustums.far[idfrustum].z,
             frustums.far[idfrustum].w);
#endif
}/*}}}*/
__host__ void create_frustums(FrustumsOri &frustums,
                              const AABB &scene_bounds,
                              const dvector<unsigned> &rays_idx,
                              const dvector<float3> &rays_ori,
                              const dvector<float3> &rays_dir,
                              const dvector<unsigned> &packet_indices,
                              const dvector<unsigned> &packet_sizes)
{
    assert(rays_ori.size() == rays_dir.size());
    assert(packet_indices.size() == packet_sizes.size());

    frustums.resize(packet_indices.size());

    dim3 dimBlock(256),
         dimGrid((frustums.size()+7)/8);

    if(dimGrid.x > 65535)
        dimGrid.x = dimGrid.y = ceil(sqrt(dimGrid.x));

    create_frustums<<<dimGrid, dimBlock>>>(frustums, scene_bounds,
                                           rays_idx, rays_ori, rays_dir, 
                                           packet_indices,
                                           packet_sizes, packet_indices.size());
}/*}}}*/

__global__ void create_mask(unsigned *mask, const int *idx,/*{{{*/
                            unsigned count)
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= count)
        return;

    mask[i] = idx[i] != -1;
}/*}}}*/

__global__ void filter_rays(int *out_idx, float3 *out_pos, float3 *out_normal,/*{{{*/
                            const int *in_idx, const float3 *in_pos,
                            const float3 *in_normal,
                            const unsigned *dest_idx, const unsigned *mask,
                            unsigned count)
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= count)
        return;

    if(mask[i])
    {
        unsigned dest = dest_idx[i];

        out_idx[dest] = in_idx[i];
        out_pos[dest] = in_pos[i];
        out_normal[dest] = in_normal[i];
    }
}/*}}}*/

__host__ void filter_rays(dvector<int> &idx, dvector<float3> &pos,/*{{{*/
                          dvector<float3> &normal)
{
    dvector<unsigned> mask;
    mask.resize(idx.size());

    dim3 dimBlock(256),
         dimGrid((idx.size()+255)/256);

    create_mask<<<dimGrid, dimBlock>>>(mask, idx, idx.size());

    dvector<unsigned> dest_idx;
    scan_add(dest_idx, mask, EXCLUSIVE);

    int len = dest_idx.back() + mask.back();

    dvector<int> out_idx;
    dvector<float3> out_pos, out_normal;
    out_idx.resize(len);
    out_pos.resize(len);
    out_normal.resize(len);

    filter_rays<<<dimGrid, dimBlock>>>(out_idx, out_pos, out_normal, 
                                       idx, pos, normal,
                                       dest_idx, mask, mask.size());

    swap(out_idx, idx);
    swap(out_pos, pos);
    swap(out_normal, normal);
}/*}}}*/

#if 0
//{{{ bvh_trace -------------------------------------------------------
__global__ void bvh_trace(float3 ray_ori,
                          const float3 *rays_dir,
                          const unsigned *rays_idx,

                          const unsigned *packet_indices,
                          const unsigned *packet_sizes,

                          const int3 *triangles,
                          const float3 *tri_vertices,
                          const float3 *tri_normals,
                          const linear_8bvh_node *bvh,
                          float4 *output)
{
    __shared__ unsigned packet_size, packet_index;

    if(threadIdx.x == 0)
    {
        packet_size = packet_sizes[blockIdx.x];
        packet_index = packet_indices[blockIdx.x];
    }
    __syncthreads();

    if(threadIdx.x >= packet_size)
        return;

    int ray_idx = packet_index + threadIdx.x;

    float3 ray_dir = rays_dir[ray_idx];

    struct stack_element
    {
        __device__ stack_element() {}
        __device__ stack_element(const linear_8bvh_node *_node, unsigned _child)
            : node(_node), next_child(_child) {}

        const linear_8bvh_node *node;
        unsigned next_child;
    };

    float3 N;
    float t = FLT_MAX;

    stack_element stack[20], *top = stack;
    *top++ = stack_element(bvh, 0);

    unsigned max_level = 0;

    while(top != stack)
    {
        stack_element *e = top-1;

        linear_8bvh_node node = *e->node;

        if(e-stack > max_level)
            max_level = e-stack;

        if(e->next_child == 0)
        {
            if(!intersects(ray_ori, ray_dir, SCENE_EPSILON, 10000,
                          node.aabb))
            {
                --top;
                continue;
            }
        }

        if(node.prim_count > 0)
        {
            unsigned tri_start = node.prim_offset;
            unsigned tri_end = tri_start + node.prim_count;
            for(unsigned i=tri_start; i<tri_end; ++i)
            {
                intersect(ray_ori, ray_dir, SCENE_EPSILON, 1000, 
                          triangles[i], tri_vertices, tri_normals, &t, NULL,&N);
            }

            --top;
            continue;
        }

        if(e->next_child == node.children_count)
        {
            --top;
            continue;
        }

        *top++ = stack_element(bvh+node.children_offset+e->next_child++,0);
    }

    float3 c = make_float3(0,0,0);

    ray_idx = rays_idx[ray_idx];

    if(t < FLT_MAX)
    {
        c = make_float3(0.1f,0.1f,0.1f);

        float3 P = ray_ori + t*ray_dir;
        for(int i=0; i<num_lights; ++i)
        {
            float kd = clamp(dot(N, lights[i].pos-P), 0.f, 1.f);

            c += make_float3(1)*kd;
        }
    }
    else
        c = temperature(max_level/6.0);
//    else if(max_level >= 1 && idisec_child >= 0)
//        c = temperature(idisec_child/7.0);

    output[ray_idx] = make_float4(c,1);//*0.5 + output[ray_idx]*0.5;

//    output[ray.pos] = make_float4(make_float3((ray.hash%521)/521.0f),1);
#if 0
    unsigned tst=blockIdx.x;
    hash_combine(tst, blockIdx.x);
    hash_combine(tst, blockIdx.x);
    output[ray.pos] = make_float4(temperature((tst%521)/521.0f),1);
#endif
}/*}}}*/
//{{{ simple_trace -------------------------------------------------------
__global__ void simple_trace(float3 ray_ori,
                             const float3 *rays_dir,
                             unsigned ray_count,
                             const int3 *triangles,
                             const float3 *tri_vertices,
                             const float3 *tri_normals,
                             unsigned tri_count,
                             float4 *output)
{
    int ray_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(ray_idx >= ray_count)
        return;

    float3 ray_dir = rays_dir[ray_idx];

    float t = 10000;
    float3 N = make_float3(0,0,0);

    for(unsigned i=0; i<tri_count; ++i)
    {
        int3 tri = triangles[i];
        N = tri_vertices[tri.x] + tri_vertices[tri.y] + tri_vertices[tri.z];
#if 0
        intersect(ray_ori, ray_dir, SCENE_EPSILON, 1000, 
                  triangles[i], tri_vertices, tri_normals, &t, NULL, &N);
#endif
    }

    float3 c = make_float3(0,0,0);

    if(t < 10000.0f)
    {
        c = make_float3(0.01f,0.01f,0.01f);

        float3 P = ray_ori + t*ray_dir;
        for(int i=0; i<num_lights; ++i)
        {
            float kd = clamp(dot(N, lights[i].pos-P), 0.f, 1.f);

            c += make_float3(1)*kd;
        }
    }

    output[ray_idx] = make_float4(c,1);
}/*}}}*/

//{{{ packet_trace -------------------------------------------------------
__global__ void packet_trace(float3 ray_ori,
                             const float3 *rays_dir,
                             const unsigned *rays_idx,
                             const unsigned *packet_indices,
                             const unsigned *packet_sizes,
                             const int3 *triangles,
                             const float3 *tri_vertices,
                             const float3 *tri_normals,
                             unsigned tri_count,
                             float4 *output)
{
    __shared__ unsigned packet_size, packet_index;

    if(threadIdx.x == 0)
    {
        packet_size = packet_sizes[blockIdx.x];
        packet_index = packet_indices[blockIdx.x];
    }
    __syncthreads();

    if(threadIdx.x >= packet_size)
        return;

    int ray_idx = packet_index + threadIdx.x;

#if 0

    float3 ray_dir = rays_dir[ray_idx];
    float t = 10000;
    float3 N;

    for(unsigned i=0; i<tri_count; ++i)
    {
        intersect(ray_ori, ray_dir, SCENE_EPSILON, 1000, 
                  triangles[i], tri_vertices, tri_normals, &t, NULL, &N);
    }

    float3 c = make_float3(0,0,0);
#endif

    ray_idx = rays_idx[ray_idx];

#if 0
    if(t < 10000.0f)
    {
        float3 P = ray_ori + t*ray_dir;
        for(int i=0; i<num_lights; ++i)
        {
            float kd = clamp(dot(N, lights[i].pos-P), 0.f, 1.f);

            c += make_float3(1)*kd;
        }
    }
#endif

    unsigned hash = 123;
    hash_combine(hash, blockIdx.x);
    hash_combine(hash, blockIdx.x);
    hash_combine(hash, blockIdx.x);
    hash_combine(hash, blockIdx.x);

    float3 c = make_float3((hash%gridDim.x) / float(gridDim.x));

    output[ray_idx] = make_float4(c,1);
}/*}}}*/
#endif

__global__ void render_frustums(float3 eye,/*{{{*/
                                float3 invU, float3 invV, float3 invW,
                                FrustumsOriGPU frustums,
                                unsigned count,
                                unsigned width, unsigned height,
                                float4 *output)
{
    int idfrustum = blockIdx.x*blockDim.x + threadIdx.x;
    if(idfrustum >= count)
        return;

    FrustumOri frustum;
    frustum.top = frustums.top[idfrustum];
    frustum.right = frustums.right[idfrustum];
    frustum.bottom = frustums.bottom[idfrustum];
    frustum.left = frustums.left[idfrustum];
    frustum.near = frustums.near[idfrustum];
    float3 ori = frustums.ori[idfrustum] - eye;

    float3 dir = cross(make_float3(frustum.top), make_float3(frustum.right)) +
                 cross(make_float3(frustum.right), make_float3(frustum.bottom))+
                 cross(make_float3(frustum.bottom), make_float3(frustum.left)) +
                 cross(make_float3(frustum.left), make_float3(frustum.top));

    dir = unit(dir);

//    cuPrintf("\t%f\t%f\t%f\n",dir.x,dir.y,dir.z);

    float3 f0 = ori,
           f1 = ori+dir/50;

    int2 p0 = project(f0, invU, invV, invW, width, height),
         p1 = project(f1, invU, invV, invW, width, height);

    float3 c = temperature(float(idfrustum) / count);

#if 0
    int idx = p1.y*width + p1.x;
    if(idx >= 0 && idx < width*height)
        output[idx] = make_float4(c, 1);

    return;
#endif

    bool steep = abs(p1.x-p0.x) < abs(p1.y-p0.y);

    if(steep)
    {
        swap(p0.x, p0.y);
        swap(p1.x, p1.y);
    }

    bool must_swap = p0.x > p1.x;
    if(must_swap)
        swap(p0, p1);

    int iy = p0.y > p1.y ? -1 : 1;

    if(p1.x == p0.x)
        p1.x++;

    float error = 0,
          edelta = abs(p1.y-p0.y) / float(p1.x-p0.x);

    while(p0.x <= p1.x)
    {
        int idx = steep ? p0.x*width + p0.y : p0.y*width + p0.x;
        if(idx >= 0 && idx < width*height)
            output[idx] = make_float4(c, 1);

        error += edelta;
        if(error >= 0.5)
        {
            p0.y += iy;
            error -= 1;
        }
        p0.x++;
    }
}/*}}}*/

//{{{ shadow_packet_trace --------------------------------------------------
__global__ void shadow_packet_trace(
                             const unsigned *on_shadow,
                             const int *rays_idx,
                             const unsigned *shadow_rays_idx,
                             const unsigned *packet_indices,
                             const unsigned *packet_sizes,
                             unsigned count,
                             float4 *output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    unsigned size = packet_sizes[idx],
             base = packet_indices[idx];

    unsigned aux = idx;
    hash_combine(aux, aux);
    hash_combine(aux, aux);
    hash_combine(aux, aux);

    float3 c = temperature((aux%1001)/1000.0);

    for(unsigned i=0; i<size; ++i)
    {
        int id = shadow_rays_idx[base+i];

//        if(!on_shadow[id])
//            continue;

        int pos = rays_idx[id];

        if(pos < 0)
            continue;

        output[pos] = make_float4(c,1);
    }
}/*}}}*/

//{{{ Create ray packets
__global__ void create_head_flags(unsigned *head_flags,/*{{{*/
                                  const unsigned *hash, unsigned count)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    if(idx > 0)
        head_flags[idx] = hash[idx] != hash[idx-1] ? 1 : 0;
    else
        head_flags[0] = 1;
}/*}}}*/
__global__ void create_comp2_size(unsigned *comp_size, /*{{{*/
                                  const unsigned *segscan,
                                  const unsigned *heads,
                                  const unsigned *pos,
                                  unsigned count)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    if(idx>0 && heads[idx])
        comp_size[pos[idx-1]-1] = segscan[idx-1];

    if(idx == count-1)
        comp_size[pos[idx]-1] = segscan[idx];
}/*}}}*/
__host__ void create_ray_packets(dvector<unsigned> &d_ray_packets,/*{{{*/
                                 dvector<unsigned> &d_ray_packet_sizes,
                                 dvector<unsigned> &d_rays_idx,
                                 const dvector<unsigned> &d_rays_hash,
                                 int bits)
{
    scoped_timer_stop sts(timers.add("Ray packets creation"));

    // calculate sorted ray indices ---------------------------------

    cuda_timer &t0 = timers.add(" - ray compression");
    dvector<unsigned> d_comp_hash, d_comp_base, d_comp_idx;
    compress_rays(d_comp_hash, d_comp_base, &d_comp_idx, d_rays_hash);
    t0.stop();
    cuda_timer &t1 = timers.add(" - ray sorting");
    dvector<unsigned> d_comp_size;
    sort_rays(d_comp_hash, d_comp_base, d_comp_size, d_comp_idx,
              d_rays_hash.size(), bits);
    t1.stop();


    cuda_timer &t2 = timers.add(" - ray decompression");
    dvector<unsigned> d_sorted_rays_idx;
    decompress_rays(d_sorted_rays_idx, 
                    d_comp_hash, d_comp_base, d_comp_size, d_comp_idx,
                    d_rays_hash.size());
    t2.stop();

    swap(d_rays_idx, d_sorted_rays_idx);

    // decompose into packets -------------------------------------

    dim3 dimBlock(256),
         dimGrid((d_comp_size.size()+255)/256);
    dvector<unsigned> d_head_flags;
    d_head_flags.resize(d_comp_hash.size());
    create_head_flags<<<dimGrid, dimBlock>>>(d_head_flags, d_comp_hash,
                                             d_comp_hash.size());

    dvector<unsigned> d_segscan;
    segscan_add(d_segscan, d_comp_size, d_head_flags);

    dvector<unsigned> d_pos;
    scan_add(d_pos, d_head_flags, INCLUSIVE);

    d_comp_size.resize(d_pos.back());
    create_comp2_size<<<dimGrid, dimBlock>>>(d_comp_size, d_segscan, 
                                             d_head_flags, d_pos,
                                             d_head_flags.size());
    d_comp_base.resize(d_comp_size.size());
    scan_add(d_comp_base, d_comp_size, EXCLUSIVE);

    cuda_timer &t3 = timers.add(" - create packets");

    decompose_into_packets(d_ray_packets, d_ray_packet_sizes, NULL,
                           d_comp_base, d_comp_size, PACKET_SIZE);

    t3.stop();
}/*}}}*/
//}}}

__global__ void soft_shadow_phong_shade(float4 *output, /*{{{*/
                      const int *idrays,
                      const float3 *positions,
                      const float3 *normals,
                      const unsigned *on_shadow,
                      unsigned count)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    int idray = idrays[idx];

    float3 c = make_float3(0,0,0);

    float3 P = positions[idx],
           N = normals[idx];
    float kd = clamp(dot(N, unit(lights[0].pos-P)), 0.f, 1.f);

    for(int i=0; i<SHADOW_TOTAL_SAMPLES; ++i)
    {
        int id = idx*SHADOW_TOTAL_SAMPLES+i;

        if(!on_shadow[id])
            c += make_float3(1)*kd;
    }

    c /= SHADOW_TOTAL_SAMPLES;

    c += make_float3(0.1,0.1,0.1);

//    c = make_float3(output[idray])*0.3 + 0.7*c;

    output[idray] = make_float4(c,1);
}

__host__ void soft_shadow_phong_shade(float4 *d_output,
                    const dvector<int> &d_idrays,
                    const dvector<float3> &d_positions,
                    const dvector<float3> &d_normals,
                    const dvector<unsigned> &d_on_shadow)
{
    assert(d_positions.size() == d_normals.size());
    assert(d_positions.size() == d_idrays.size());
    assert(d_positions.size()*SHADOW_SAMPLES == d_on_shadow.size());

    dim3 dimBlock(256),
         dimGrid((d_positions.size()+255)/256);

    scoped_timer_stop sts(timers.add("Phong shading"));

    soft_shadow_phong_shade<<<dimGrid, dimBlock>>>(d_output,
                                       d_idrays, d_positions, d_normals,
                                       d_on_shadow,
                                       d_positions.size());
}/*}}}*/

__global__ void phong_shade(float4 *output, /*{{{*/
                      const int *idrays,
                      const float3 *positions,
                      const float3 *normals,
                      unsigned count)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    int idray = idrays[idx];
    if(idray < 0)
        return;

    float3 c = make_float3(0.1,0.1,0.1);

    float3 P = positions[idx],
           N = normals[idx];
    for(int i=0; i<num_lights; ++i)
    {
        float kd = clamp(dot(N, unit(lights[i].pos-P)), 0.f, 1.f);

        c += make_float3(1)*kd;
    }

    output[idray] = make_float4(c,1);
}

__host__ void phong_shade(float4 *d_output,
                    const dvector<int> &d_idrays,
                    const dvector<float3> &d_positions,
                    const dvector<float3> &d_normals)
{
    assert(d_positions.size() == d_normals.size());
    assert(d_positions.size() == d_idrays.size());

    dim3 dimBlock(256),
         dimGrid((d_idrays.size()+255)/256);

    scoped_timer_stop sts(timers.add("Phong shading"));

    phong_shade<<<dimGrid, dimBlock>>>(d_output,
                                       d_idrays, d_positions, d_normals,
                                       d_idrays.size());
}/*}}}*/

__host__ void cuda_trace(float3 U, float3 V, float3 W, 
                         float3 invU, float3 invV, float3 invW, 
                         float3 eye,
                         const Mesh &mesh, 
                         const std::vector<linear_8bvh_node> &bvh,
                         size_t bvh_height,
                         cudaGraphicsResource *output,
                         int width, int height,
                         bool recalc_rays,
                         bool reload_model)
{
    static bool init=false;
    if(!init)
    {
        cudaPrintfInit(10*1024*1024);
        init = true;
    }

    dvector<float3> d_rays_dir;
    dvector<unsigned> d_rays_idx;
    dvector<unsigned> d_rays_hash;

    Frustums d_frustums;

    static dvector<float4> d_tri_xform;
    static dvector<float4> d_tri_normals; // soh uso xyz
    static bvh_soa d_bvh;

    dvector<unsigned> d_ray_packets, d_ray_packet_sizes;

    if(reload_model || d_tri_xform.empty())
    {
        d_tri_xform = mesh.xform;
        d_tri_normals.resize(mesh.normals.size());

        cudaMemcpy2D(d_tri_normals, sizeof(float4), 
                     &mesh.normals[0], sizeof(float3), 
                     sizeof(float3),mesh.normals.size(),cudaMemcpyHostToDevice);

        assert(d_tri_xform.size() == d_tri_normals.size());

        convert_to_soa(d_bvh, bvh);

        std::cout << "BVH height: " << bvh_height << std::endl;
        std::cout << "Triangles: " << d_tri_xform.size()/3 << std::endl;
        std::cout << "Nodes: " << bvh.size() << std::endl;
    }

    timers.add("Total time");

    {
        d_rays_hash.resize(width*height);

        float cell_size = 
            create_primary_rays(d_rays_dir, d_rays_hash, bvh[0].aabb, U, V, W,
                                width, height);

        int bits = (floor(log2(1/cell_size)+1))*2;

        create_ray_packets(d_ray_packets, d_ray_packet_sizes, d_rays_idx,
                           d_rays_hash, bits);

        create_frustums(d_frustums, eye, d_rays_idx, d_rays_dir, 
                        d_ray_packets, d_ray_packet_sizes);
    }

    cudaGraphicsMapResources(1, &output, 0);
    check_cuda_error("Error mapping output buffer");

    float4 *d_output;
    size_t output_size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &output_size, output);
    check_cuda_error("Error getting output pointer");

    assert(output_size == width*height*sizeof(float4));

//    if(false)
    {
        scoped_timer_stop sts(timers.add("Screen buffer clearing"));
        cudaMemset(d_output, 0, output_size);
    }

//    cuda_timer timer;

    dvector<float3> d_isect_positions;
    dvector<float3> d_isect_normals;
    dvector<int> d_isect_rays_idx;

    {
        dvector<unsigned> d_active_ray_packets, 
                          d_active_ray_packet_sizes,
                          d_active_frustum_leaves,
                          d_active_frustum_leaf_sizes,
                          d_active_idleaves;

        traverse(d_active_ray_packets, d_active_ray_packet_sizes,
                 d_active_frustum_leaves, d_active_frustum_leaf_sizes,
                 d_active_idleaves,
                 bvh[0].children_count, d_bvh, bvh_height,
                 d_frustums, d_ray_packets, d_ray_packet_sizes);


        primary_local_intersections(
            d_isect_positions, d_isect_normals, d_isect_rays_idx,
            d_active_ray_packets, d_active_ray_packet_sizes,
            d_active_frustum_leaves, d_active_frustum_leaf_sizes,
            d_active_idleaves, d_bvh,
            d_tri_xform, d_tri_normals,
            eye, d_rays_dir, d_rays_idx);

        filter_rays(d_isect_rays_idx, d_isect_positions, d_isect_normals);
    }

    dvector<float3> d_shadow_rays_ori, d_shadow_rays_dir;
    float cell_size = create_soft_shadow_rays(d_shadow_rays_ori,
                                              d_shadow_rays_dir, 
                                              d_rays_hash, 
                       bvh[0].aabb, d_isect_positions);

//    cudaPrintfDisplay(stdout, false);
//    exit(0);

    dvector<unsigned> d_shadow_ray_packets, d_shadow_ray_packet_sizes,
                             d_shadow_rays_idx;

    int bits = 32;//ceil(log2(1/cell_size))*3;

//    std::cout << "---------------------------------------------------" << std::endl;

    create_ray_packets(d_shadow_ray_packets, d_shadow_ray_packet_sizes,
                       d_shadow_rays_idx, d_rays_hash, bits);
#if 0
    std::vector<float> h_ray_pdf = to_cpu(d_shadow_rays_pdf);
    for(int i=0; i<h_ray_pdf.size(); ++i)
    {
        std::cout << i << ": " << h_ray_pdf[i] << std::endl;
    }

    exit(0);
#endif

#if 0
    std::vector<float3> aux = to_cpu(d_isect_positions);
    for(int i=0; i<aux.size(); ++i)
        printf("%f\t%f\t%f\n",aux[i].x,aux[i].y,aux[i].z);
    exit(0);

    std::vector<float3> aux1 = to_cpu(d_shadow_rays_ori),
                        aux2 = to_cpu(d_isect_positions);
    for(int i=0; i<aux1.size(); ++i)
        std::cout << aux1[i] << " - " << aux2[i] << std::endl;
    exit(0);
#endif


    FrustumsOri d_shadow_frustums;
    create_frustums(d_shadow_frustums, bvh[0].aabb,
                    d_shadow_rays_idx, d_shadow_rays_ori, d_shadow_rays_dir, 
                    d_shadow_ray_packets, d_shadow_ray_packet_sizes);

#if TRACE_FRUSTUM_SIZE
    cudaPrintfDisplay(stdout, false);
    exit(0);
#endif


#if 0
    dim3 dimBlock(256),
         dimGrid((d_shadow_frustums.size()+255)/256);
    render_frustums<<<dimGrid,dimBlock>>>(eye, invU, invV, invW, 
                                          d_shadow_frustums,
                                          d_shadow_frustums.size(),
                                          width, height, d_output);

#endif

#if 0
    std::vector<float4> h_top = to_cpu(d_shadow_frustums.top),
                        h_left = to_cpu(d_shadow_frustums.left),
                        h_bottom = to_cpu(d_shadow_frustums.bottom),
                        h_right = to_cpu(d_shadow_frustums.right),
                        h_near = to_cpu(d_shadow_frustums.near);

    for(int i=0; i<h_top.size(); ++i)
    {
        std::cout << "Frustum " << i << " ------------------------\n";
        std::cout << "    top: " << h_top[i] << '\n'
                  << "  right: " << h_right[i] << '\n'
                  << " bottom: " << h_bottom[i] << '\n'
                  << "   left: " << h_left[i] << '\n'
                  << "   near: " << h_near[i] << std::endl;
    }
    exit(0);
#endif

#if 1

    dvector<unsigned> d_active_ray_packets, 
                      d_active_ray_packet_sizes,
                      d_active_frustum_leaves,
                      d_active_frustum_leaf_sizes,
                      d_active_idleaves;

    traverse(d_active_ray_packets, d_active_ray_packet_sizes,
             d_active_frustum_leaves, d_active_frustum_leaf_sizes,
             d_active_idleaves,
             bvh[0].children_count, d_bvh, bvh_height,
             d_shadow_frustums, d_shadow_ray_packets,d_shadow_ray_packet_sizes);

    if(!d_active_ray_packets.empty())
    {
        dvector<unsigned> d_on_shadow;

        shadow_local_intersections(d_on_shadow,
                            d_active_ray_packets, d_active_ray_packet_sizes,
                            d_active_frustum_leaves,d_active_frustum_leaf_sizes,
                            d_active_idleaves,
                            d_bvh,
                            d_tri_xform,
                            d_shadow_rays_ori,d_shadow_rays_dir,
                            d_shadow_rays_idx);

        soft_shadow_phong_shade(d_output,
                    d_isect_rays_idx, d_isect_positions, d_isect_normals,
                    d_on_shadow);
    }


#endif

    timers.flush();



#if 0
    static int cnt=0;
    if(cnt++ == 50)
        exit(0);
#endif
//    exit(0);

    // this is important!! without it, somehow, FPS drops 21%
    cudaThreadSynchronize();

    cudaGraphicsUnmapResources(1, &output, 0);
    check_cuda_error("Error unmapping output buffer");
}

