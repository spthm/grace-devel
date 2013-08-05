#ifndef FRUSTUM_H
#define FRUSTUM_H

#include "util.h"
#include "dvector.h"
#include "aabb.h"
#include "plane.h"
#include <cassert>


struct Frustum
{
    float4 top, right, bottom, left;
    unsigned dirsign; 
};

struct const_FrustumsGPU
{
    unsigned size;
    const float4 *top, *right, *bottom, *left;
    const char *dirsign; 

    typedef Frustum frustum_type;

    __device__ frustum_type to_frustum(int idfrustum) const
    {
        frustum_type f;
        f.top = top[idfrustum];
        f.right = right[idfrustum];
        f.bottom = bottom[idfrustum];
        f.left = left[idfrustum];
        f.dirsign = dirsign[idfrustum];
        return f;
    }
};

struct FrustumsGPU
{
    unsigned size;
    float4 *top, *right, *bottom, *left;
    char *dirsign; 

    typedef Frustum frustum_type;

    __device__ frustum_type to_frustum(int idfrustum) const
    {
        frustum_type f;
        f.top = top[idfrustum];
        f.right = right[idfrustum];
        f.bottom = bottom[idfrustum];
        f.left = left[idfrustum];
        f.dirsign = dirsign[idfrustum];
        return f;
    }

    operator const_FrustumsGPU &() const
    {
        return *reinterpret_cast<const_FrustumsGPU *>(const_cast<FrustumsGPU *>(this));
    }
};

struct Frustums
{
    static const bool has_near = false;

    typedef FrustumsGPU gpu_type;
    typedef const_FrustumsGPU const_gpu_type;

    dvector<float4> top, right, bottom, left;
    dvector<char> dirsign;

    size_t size() const { return dirsign.size(); }

    __host__ void resize(size_t s)/*{{{*/
    {
        top.resize(s);
        right.resize(s);
        bottom.resize(s);
        left.resize(s);
        dirsign.resize(s);
    }/*}}}*/

    __host__ operator const_FrustumsGPU() const/*{{{*/
    {
        assert(top.size() == right.size());
        assert(top.size() == bottom.size());
        assert(top.size() == left.size());
        assert(top.size() == dirsign.size());

        const_FrustumsGPU f;
        f.size = top.size();
        f.top = top;
        f.right = right;
        f.bottom = bottom;
        f.left = left;
        f.dirsign = dirsign;

        return f;
    }/*}}}*/

    __host__ operator FrustumsGPU()/*{{{*/
    {
        assert(top.size() == right.size());
        assert(top.size() == bottom.size());
        assert(top.size() == left.size());
        assert(top.size() == dirsign.size());

        FrustumsGPU f;
        f.size = top.size();
        f.top = top;
        f.right = right;
        f.bottom = bottom;
        f.left = left;
        f.dirsign = dirsign;

        return f;
    }/*}}}*/
};

struct FrustumOri
{
    float4 top, right, bottom, left, near, far;
    unsigned dirsign; 
};

struct const_FrustumsOriGPU
{
    unsigned size;
    const float4 *top, *right, *bottom, *left, *near, *far;
    const float3 *ori;
    const char *dirsign; 

    typedef FrustumOri frustum_type;

    __device__ frustum_type to_frustum(int idfrustum) const
    {
        frustum_type f;
        f.top = top[idfrustum];
        f.right = right[idfrustum];
        f.bottom = bottom[idfrustum];
        f.left = left[idfrustum];
        f.near = near[idfrustum];
        f.far = far[idfrustum];
        f.dirsign = dirsign[idfrustum];
        return f;
    }
};

struct FrustumsOriGPU
{
    unsigned size;
    float4 *top, *right, *bottom, *left, *near, *far;
    char *dirsign; 
    float3 *ori;

    typedef FrustumOri frustum_type;

    __device__ frustum_type to_frustum(int idfrustum) const
    {
        frustum_type f;
        f.top = top[idfrustum];
        f.right = right[idfrustum];
        f.bottom = bottom[idfrustum];
        f.left = left[idfrustum];
        f.near = near[idfrustum];
        f.far = far[idfrustum];
        f.dirsign = dirsign[idfrustum];
        return f;
    }

    operator const_FrustumsOriGPU &() const
    {
        return *reinterpret_cast<const_FrustumsOriGPU *>(const_cast<FrustumsOriGPU *>(this));
    }
};

struct FrustumsOri
{
    static const bool has_near = true;

    typedef FrustumsOriGPU gpu_type;
    typedef const_FrustumsOriGPU const_gpu_type;

    dvector<float4> top, right, bottom, left, near, far;
    dvector<char> dirsign;

    size_t size() const { return dirsign.size(); }

    __host__ void resize(size_t s)/*{{{*/
    {
        top.resize(s);
        right.resize(s);
        bottom.resize(s);
        left.resize(s);
        near.resize(s);
        far.resize(s);
        dirsign.resize(s);
    }/*}}}*/

    __host__ operator const_FrustumsOriGPU() const/*{{{*/
    {
        assert(top.size() == right.size());
        assert(top.size() == bottom.size());
        assert(top.size() == left.size());
        assert(top.size() == near.size());
        assert(top.size() == dirsign.size());
        assert(top.size() == far.size());

        const_FrustumsOriGPU f;
        f.size = top.size();
        f.top = top;
        f.right = right;
        f.bottom = bottom;
        f.left = left;
        f.near = near;
        f.far = far;
        f.dirsign = dirsign;

        return f;
    }/*}}}*/

    __host__ operator FrustumsOriGPU()/*{{{*/
    {
        assert(top.size() == right.size());
        assert(top.size() == bottom.size());
        assert(top.size() == left.size());
        assert(top.size() == near.size());
        assert(top.size() == dirsign.size());
        assert(top.size() == far.size());

        FrustumsOriGPU f;
        f.size = top.size();
        f.top = top;
        f.right = right;
        f.bottom = bottom;
        f.left = left;
        f.near = near;
        f.far = far;
        f.dirsign = dirsign;

        return f;
    }/*}}}*/
};


inline __host__ __device__ 
bool intersects(const AABB &aabb, const Frustum &frustum)/*{{{*/
{
    if(!intersects(aabb, frustum.top))
        return false;

    if(!intersects(aabb, frustum.right))
        return false;

    if(!intersects(aabb, frustum.bottom))
        return false;

    if(!intersects(aabb, frustum.left))
        return false;

    return true;
}/*}}}*/

inline __host__ __device__ 
bool intersects(const AABB &aabb, const const_FrustumsGPU &frustum, unsigned id)/*{{{*/
{
    if(!intersects(aabb, frustum.top[id]))
        return false;

    if(!intersects(aabb, frustum.right[id]))
        return false;

    if(!intersects(aabb, frustum.bottom[id]))
        return false;

    if(!intersects(aabb, frustum.left[id]))
        return false;

    return true;
}/*}}}*/

inline __host__ __device__ 
bool intersects(const AABB &aabb, const FrustumOri &frustum)/*{{{*/
{
    if(!intersects(aabb, frustum.top))
        return false;

    if(!intersects(aabb, frustum.right))
        return false;

    if(!intersects(aabb, frustum.bottom))
        return false;

    if(!intersects(aabb, frustum.left))
        return false;

    if(!intersects(aabb, frustum.far))
        return false;

    if(!intersects(aabb, frustum.near))
        return false;

    return true;
}/*}}}*/

inline __host__ __device__ 
bool intersects(const AABB &aabb, const const_FrustumsOriGPU &frustum, unsigned id)/*{{{*/
{
    if(!intersects(aabb, frustum.top[id]))
        return false;

    if(!intersects(aabb, frustum.right[id]))
        return false;

    if(!intersects(aabb, frustum.bottom[id]))
        return false;

    if(!intersects(aabb, frustum.left[id]))
        return false;

    if(!intersects(aabb, frustum.far[id]))
        return false;

    if(!intersects(aabb, frustum.near[id]))
        return false;

    return true;
}/*}}}*/

#endif
