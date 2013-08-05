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
#include "bvh.h"
#include "triangle.cu"
#include "timer.h"
#include "traversal.h"
#include "cuda_bvh.h"

#define PACKET_SIZE 256

#define TRACE 0

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

    // [0,col],[0,row] -> [-1,1],[-1,1]
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
    dim3 dimGrid((width + dimBlock.x-1)/dimBlock.x,
                 (height + dimBlock.y-1)/dimBlock.y);

    d_rays_dir.resize(width*height);
    d_rays_hash.resize(width*height);

    float cell_size = length(scene_bounds.hsize)*2 * 0.004;

    scoped_timer_stop sts(timers.add("Primary ray creation"));

    create_primary_rays<<<dimGrid,dimBlock>>>(U,V,W,width,height,
                                              1.0f/cell_size,
                                      d_rays_dir, d_rays_hash);

    return cell_size;
}
/*}}}*/

//{{{ shuffle -----------------------------------------------------------------
__global__ void reorder_rays(const unsigned *indices,
                             const float3 *input_ray_dir,
                             const unsigned *input_ray_hash,
                             float3 *output_ray_dir, 
                             unsigned *output_ray_hash, 
                             size_t count)
{
    int dest_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(dest_idx >= count)
        return;

    int src_idx = indices[dest_idx];

    output_ray_dir[dest_idx] = input_ray_dir[src_idx];
    output_ray_hash[dest_idx] = input_ray_hash[src_idx];
}

__host__ void reorder_rays(dvector<float3> &output_ray_dir,
                           dvector<unsigned> &output_ray_hash,
                           const dvector<float3> &input_ray_dir,
                           const dvector<unsigned> &input_ray_hash,
                           const dvector<unsigned> &indices)
{
    output_ray_dir.resize(input_ray_dir.size());
    output_ray_hash.resize(input_ray_hash.size());

    dim3 dimGrid, dimBlock;
    compute_linear_grid(input_ray_dir.size(), dimGrid, dimBlock);

    reorder_rays<<<dimGrid, dimBlock>>>(
        indices, input_ray_dir, input_ray_hash,
                 output_ray_dir, output_ray_hash,
        input_ray_dir.size());
}/*}}}*/

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
        frustums.top[idfrustum]    = create_plane(O, z_axis, -v_max,0,1);
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

__device__ 
float4 create_plane(const float3 &o, const float3 &d0, const float3 &d1)/*{{{*/
{
    float3 n = unit(cross(d0, d1));
    return make_float4(n, -dot(n,o));
}/*}}}*/

__global__ void create_frustums(FrustumOri *frustums,/*{{{*/
                                const AABB scene_bounds,
                                const float3 *rays_ori,
                                const float3 *rays_dir,
                                const unsigned *packet_indices,
                                const unsigned *packet_sizes,
                                unsigned num_packets)
{
    // 32 threads vao processar um frustum
    // cada frustum tem no máximo 256 raios, portanto cada thread vai processar
    // no máximo 8 raios.

    int tx = threadIdx.x, bx = blockIdx.x;

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

    int z_axis = dominant_axis(rays_dir[idfirst_ray]);

    // posicao do plano near (bounds das origens na direcao -z_axis)

    float near_plane;
    switch(z_axis)
    {
    case 1: // x-axis is dominant
        near_plane = FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;
            float ori = rays_ori[idfirst_ray + idray + i].x;
            near_plane = min(near_plane, ori);
        }
        reduce_min(idray, tx, near_plane, buffer);
        break;
    case -1: // -x-axis is dominant
        near_plane = -FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;
            float ori = rays_ori[idfirst_ray + idray + i].x;
            near_plane = max(near_plane, ori);
        }
        reduce_max(idray, tx, near_plane, buffer);
        break;
    case 2: // y-axis is dominant
        near_plane = FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;
            float ori = rays_ori[idfirst_ray + idray + i].y;
            near_plane = min(near_plane, ori);
        }
        reduce_min(idray, tx, near_plane, buffer);
        break;
    case -2: // -y-axis is dominant
        near_plane = -FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;
            float ori = rays_ori[idfirst_ray + idray + i].y;
            near_plane = max(near_plane, ori);
        }
        reduce_max(idray, tx, near_plane, buffer);
        break;
    case 3: // z-axis is dominant
        near_plane = FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;
            float ori = rays_ori[idfirst_ray + idray + i].z;
            near_plane = min(near_plane, ori);
        }
        reduce_min(idray, tx, near_plane, buffer);
        break;
    case -3: // -z-axis is dominant
        near_plane = -FLT_MAX;
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;
            float ori = rays_ori[idfirst_ray + idray + i].z;
            near_plane = max(near_plane, ori);
        }
        reduce_max(idray, tx, near_plane, buffer);
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

            unsigned idrayabs = idfirst_ray + idray + i;

            float ori = rays_ori[idrayabs].x;

            float3 dir = rays_dir[idrayabs];
            float inv = 1.0f/dir.x;

            float fp = (far_plane-ori)*inv;
            min_max(far_u_min, far_u_max, dir.y * fp);
            min_max(far_v_min, far_v_max, dir.z * fp);

            float np = (near_plane-ori)*inv;
            min_max(near_u_min, near_u_max, dir.y * np);
            min_max(near_v_min, near_v_max, dir.z * np);
        }
        break;
    case 2: // y-axis is dominant
        far_plane = scene_bounds.center.y + copysignf(scene_bounds.hsize.y,z_axis);
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            unsigned idrayabs = idfirst_ray + idray + i;

            float ori = rays_ori[idrayabs].y;

            float3 dir = rays_dir[idrayabs];
            float inv = 1.0f/dir.y;

            float fp = (far_plane-ori)*inv;
            min_max(far_u_min, far_u_max, dir.x * fp);
            min_max(far_v_min, far_v_max, dir.z * fp);

            float np = (near_plane-ori)*inv;
            min_max(near_u_min, near_u_max, dir.x * np);
            min_max(near_v_min, near_v_max, dir.z * np);
        }
        break;
    case 3: // z-axis is dominant
        far_plane = scene_bounds.center.z + copysignf(scene_bounds.hsize.z,z_axis);
        for(int i=0; i<256; i+=32)
        {
            if(idray+i >= packet_size)
                break;

            unsigned idrayabs = idfirst_ray + idray + i;

            float ori = rays_ori[idrayabs].z;

            float3 dir = rays_dir[idrayabs];
            float inv = 1.0f/dir.z;

            float fp = abs(far_plane-ori)*inv;
            min_max(far_u_min, far_u_max, dir.x * fp);
            min_max(far_v_min, far_v_max, dir.y * fp);

            float np = (near_plane-ori)*inv;
            min_max(near_u_min, near_u_max, dir.x * np);
            min_max(near_v_min, near_v_max, dir.y * np);
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

    if(abs(far_u_max-far_u_min) < 1e-5f)
    {
        far_u_max = far_u_min+1e-5f;
        far_u_min -= 1e-5f;
    }
    if(abs(far_v_max - far_v_min) < 1e-5f)
    {
        far_v_max = far_v_min+1e-5f;
        far_v_min -= 1e-5f;
    }

    FrustumOri &frustum = frustums[idfrustum];


    float3 o0,o1,o2,o3, d0, d1, d2, d3;
    float3 on, dn;
    switch(abs(z_axis))
    {
    case 1: // X is major axis, right is -Y, top is Z
        // top left
        o0 = make_float3(near_plane, near_u_max, near_v_max);
        d0 = make_float3(far_plane, far_u_max, far_v_max)-o0;

        // bottom left
        o1 = make_float3(near_plane, near_u_max, near_v_min);
        d1 = make_float3(far_plane, far_u_max, far_v_min)-o1;

        // bottom right
        o2 = make_float3(near_plane, near_u_min, near_v_min);
        d2 = make_float3(far_plane, far_u_min, far_v_min)-o2;;

        // top right
        o3 = make_float3(near_plane, near_u_min, near_v_max);
        d3 = make_float3(far_plane, far_u_min, far_v_max)-o3;

        on = make_float3(near_plane, (near_u_max-near_u_min)/2,
                                     (near_v_max-near_v_min)/2);
        dn = make_float3(copysignf(1,z_axis), 0, 0);
        break;
    case 2: // Y is major axis, right is +X, top is +Z
        // top left
        o0 = make_float3(near_u_min, near_plane, near_v_max);
        d0 = make_float3(far_u_min, far_plane, far_v_max)-o0;

        // bottom left
        o1 = make_float3(near_u_min, near_plane, near_v_min);
        d1 = make_float3(far_u_min, far_plane, far_v_min)-o1;

        // bottom right
        o2 = make_float3(near_u_max, near_plane, near_v_min);
        d2 = make_float3(far_u_max, far_plane, far_v_min)-o2;;

        // top right
        o3 = make_float3(near_u_max, near_plane, near_v_max);
        d3 = make_float3(far_u_max, far_plane, far_v_max)-o3;

        on = make_float3((near_u_max-near_u_min)/2, near_plane,
                         (near_v_max-near_v_min)/2);
        dn = make_float3(0,copysignf(1,z_axis), 0);
        break;
    case 3: // Z is major axis, right is -X, top is +Y
        // top left
        o0 = make_float3(near_u_max, near_v_max, near_plane);
        d0 = make_float3(far_u_max, far_v_max, far_plane)-o0;

        // bottom left
        o1 = make_float3(near_u_max, near_v_min, near_plane);
        d1 = make_float3(far_u_max, far_v_min, far_plane)-o1;

        // bottom right
        o2 = make_float3(near_u_min, near_v_min, near_plane);
        d2 = make_float3(far_u_min, far_v_min, far_plane)-o2;;

        // top right
        o3 = make_float3(near_u_min, near_v_max, near_plane);
        d3 = make_float3(far_u_min, far_v_max, far_plane)-o3;

        on = make_float3((near_u_max-near_u_min)/2, (near_v_max-near_v_min)/2,
                         near_plane);
        dn = make_float3(0,0,copysignf(1,z_axis));
        break;
    }

    if(z_axis < 0)
    {
        frustum.top = create_plane(o0, d0, d3);
        frustum.left = create_plane(o3, d3, d2);
        frustum.bottom = create_plane(o2, d2, d1);
        frustum.right = create_plane(o1, d1, d0);
    }
    else
    {
        frustum.top = create_plane(o3, d3, d0);
        frustum.left = create_plane(o0, d0, d1);
        frustum.bottom = create_plane(o1, d1, d2);
        frustum.right = create_plane(o2, d2, d3);
    }

    frustum.near = create_plane(on, 0, dn.x, dn.y, dn.z);

    frustum.dirsign = calc_dirsign(z_axis, d0.x, d0.y, d0.z);
}/*}}}*/
__host__ void create_frustums(dvector<FrustumOri> &frustums,
                              const AABB &scene_bounds,
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

    create_frustums<<<dimGrid, dimBlock>>>(frustums, scene_bounds,
                                           rays_ori, rays_dir, 
                                           packet_indices,
                                           packet_sizes, packet_indices.size());
}/*}}}*/

__global__ void create_head_flags(unsigned *head_flags,
                                  const unsigned *hash, unsigned count)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    if(idx > 0)
        head_flags[idx] = hash[idx] != hash[idx-1] ? 1 : 0;
    else
        head_flags[0] = 1;
}

__global__ void create_comp2_size(unsigned *comp_size, 
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
}

__host__ void create_ray_packets(dvector<unsigned> &d_ray_packets,/*{{{*/
                                 dvector<unsigned> &d_ray_packet_sizes,
                                 dvector<unsigned> &d_rays_idx,
                                 const dvector<unsigned> &d_rays_hash,
                                 dvector<float3> &d_rays_dir, int bits)
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

    int len = d_pos.back() + d_head_flags.back();

    d_comp_size.resize(len);
    create_comp2_size<<<dimGrid, dimBlock>>>(d_comp_size, d_segscan, 
                                             d_head_flags, d_pos,
                                             d_head_flags.size());

    d_comp_base.resize(len);
    scan_add(d_comp_base, d_comp_size, EXCLUSIVE);

    cuda_timer &t3 = timers.add(" - create packets");

    decompose_into_packets(d_ray_packets, d_ray_packet_sizes, NULL,
                           d_comp_base, d_comp_size, PACKET_SIZE);

    t3.stop();
}/*}}}*/

__host__ void cuda_trace(float3 U, float3 V, float3 W, 
                         float3 invU, float3 invV, float3 invW, 
                         float3 eye,
                         const Mesh &mesh, 
                         const std::vector<linear_8bvh_node> &bvh,
                         size_t bvh_height,
                         cudaGraphicsResource *output,
                         int width, int height, bool recalc_rays,
                         bool reload_model)
{

    static dvector<float3> d_rays_dir;
    static dvector<unsigned> d_rays_idx;
    static dvector<unsigned> d_rays_hash;

    static Frustums d_frustums;

    static dvector<float4> d_tri_xform;
    static dvector<float4> d_tri_normals; // soh uso xyz
    static bvh_soa d_bvh;

    static dvector<unsigned> d_ray_packets, d_ray_packet_sizes;

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

//    if(d_rays_dir.empty() || recalc_rays || 
//       screen_width != width || screen_height != height)
    {
        d_rays_hash.resize(width*height);

        float cell_size = 
            create_primary_rays(d_rays_dir, d_rays_hash, bvh[0].aabb, U, V, W,
                                width, height);

        int bits = ceil(log2(1/cell_size))*2;

        create_ray_packets(d_ray_packets, d_ray_packet_sizes, d_rays_idx,
                           d_rays_hash, d_rays_dir, bits);

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

    {
        scoped_timer_stop sts(timers.add("Screen buffer clearing"));
        cudaMemset(d_output, 0, output_size);
    }

#if 0
    if(!d_ray_packets.empty())
    {
        dim3 dimBlock(256), dimGrid(d_ray_packets.size());
        bvh_trace<<<dimGrid,dimBlock>>>(eye, d_rays_dir, d_rays_idx,
                                        d_ray_packets, d_ray_packet_sizes,
                                        d_triangles, d_tri_vertices, d_tri_normals,
                                        d_bvh, d_output);
    }
#endif

//    cuda_timer timer;

    static dvector<unsigned> d_active_ray_packets, 
                             d_active_ray_packet_sizes,
                             d_active_frustum_leaves,
                             d_active_frustum_leaf_sizes,
                             d_active_idnodes;

    traverse(d_active_ray_packets, d_active_ray_packet_sizes,
             d_active_frustum_leaves, d_active_frustum_leaf_sizes,
             d_active_idnodes,
             bvh[0].children_count, d_bvh, bvh_height,
             d_frustums, d_ray_packets, d_ray_packet_sizes);


    primary_local_intersections(d_output,
        d_active_ray_packets, d_active_ray_packet_sizes,
        d_active_frustum_leaves, d_active_frustum_leaf_sizes,
        d_active_idnodes, d_bvh,
        d_tri_xform, d_tri_normals,
        eye, d_rays_dir, d_rays_idx);

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

