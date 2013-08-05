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

struct FrustumOri
{
    float4 top, right, bottom, left, near;
    unsigned dirsign; 
};

struct const_FrustumsGPU
{
    unsigned size;
    const float4 *top, *right, *bottom, *left;
    const char *dirsign; 
};

struct FrustumsGPU
{
    unsigned size;
    float4 *top, *right, *bottom, *left;
    char *dirsign; 

    operator const_FrustumsGPU &() const
    {
        return *reinterpret_cast<const_FrustumsGPU *>(const_cast<FrustumsGPU *>(this));
    }
};

struct Frustums
{
    dvector<float4> top, right, bottom, left;
    dvector<char> dirsign;

    size_t size() const { return dirsign.size(); }

    __host__ void resize(size_t s)
    {
        top.resize(s);
        right.resize(s);
        bottom.resize(s);
        left.resize(s);
        dirsign.resize(s);
    }

    __host__ operator const_FrustumsGPU() const
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
    }

    __host__ operator FrustumsGPU()
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
    }
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

    if(!intersects(aabb, frustum.near))
        return false;

    return true;
}/*}}}*/

#endif
