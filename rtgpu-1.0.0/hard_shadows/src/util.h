#ifndef RTGPU_UTIL_H
#define RTGPU_UTIL_H

#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <optix_math.h>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

template <class T> class dvector;
struct linear_8bvh_node;
struct linear_8bvh_leaf;

void check_glerror();

void check_cuda_error(const std::string &msg="");

enum scan_type
{
    INCLUSIVE,
    EXCLUSIVE
};

void scan_add(dvector<unsigned> &dest, 
              const dvector<unsigned> &orig, scan_type type);

void segscan_add(dvector<unsigned> &dest, 
                 const dvector<unsigned> &orig,
                 const dvector<unsigned> &head_flags);

void sort(dvector<unsigned> &keys, dvector<unsigned> &values, int bits);

void cuda_synchronize();

void adjacent_difference(dvector<unsigned> &output, 
                         const dvector<unsigned> &input, 
                         size_t input_item_count);

const float CELLS = 80;

inline __host__ __device__ int quantize(float ang, float min, float inv_len, float cells)
{
    return (ang-min)*inv_len*cells;
}

inline __host__ __device__ void hash_combine(unsigned &seed, int val)
{
    // from boost::hash_combine
    seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

#ifdef __CUDACC__
inline __host__ __device__ 
float3 unit(float3 v)
{
    return v * rsqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}


template <class T>
inline __device__ void swap(T &a, T &b)/*{{{*/
{
    // it's faster to do like this than using xor hacks

    T aux = a;
    a = b;
    b = aux;
}/*}}}*/
#endif

inline __host__ __device__ 
int2 project(float3 pt, float3 invU, float3 invV, float3 invW,/*{{{*/
                          int width, int height)
{
    float3 pos = invU*pt.x + invV*pt.y + pt.z*invW;

//    if(pos.z <= 0)
//        return make_int2(INT_MAX, INT_MAX);

    float2 pos2d = make_float2(pos.x/pos.z, pos.y/pos.z);

    return make_int2(round((pos2d.x+1)/2*width), round((pos2d.y+1)/2*height));
}/*}}}*/

void compute_linear_grid(unsigned size, dim3 &grid, dim3 &block);

inline std::ostream &operator<<(std::ostream &out, float4 val)
{
    return out << '(' << val.x 
               << ',' << val.y 
               << ',' << val.z 
               << ',' << val.w << ')';
}

inline std::ostream &operator<<(std::ostream &out, float3 val)
{
    return out << '(' << val.x << ',' << val.y << ',' << val.z << ')';
}

inline std::ostream &operator<<(std::ostream &out, float2 val)
{
    return out << '(' << val.x << ',' << val.y << ')';
}

inline std::ostream &operator<<(std::ostream &out, dim3 val)
{
    return out << '[' << val.x << 'x' << val.y << 'x' << val.z << ']';
}

#endif
