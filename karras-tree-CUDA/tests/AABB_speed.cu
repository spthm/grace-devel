#include <sstream>
#include <cmath>

#include <assert.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/morton.cuh"
#include "../kernel_config.h"

enum DIR_CLASS
{ MMM, PMM, MPM, PPM, MMP, PMP, MPP, PPP };

struct Ray
{
    float dx, dy, dz;
    float ox, oy, oz;
    float length;
    unsigned int dclass;
};

struct RaySlope
{
    float xbyy, ybyx, ybyz, zbyy, xbyz, zbyx;
    float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
};

__host__ __device__ RaySlope ray_slope(const Ray& ray)
{
    RaySlope slope;

    slope.xbyy = ray.dx / ray.dy;
    slope.ybyx = 1.0f / slope.xbyy;
    slope.ybyz = ray.dy / ray.dz;
    slope.zbyy = 1.0f / slope.ybyz;
    slope.xbyz = ray.dx / ray.dz;
    slope.zbyx = 1.0f / slope.xbyz;

    slope.c_xy = ray.oy - slope.ybyx*ray.ox;
    slope.c_xz = ray.oz - slope.zbyx*ray.ox;
    slope.c_yx = ray.ox - slope.xbyy*ray.oy;
    slope.c_yz = ray.oz - slope.zbyy*ray.oy;
    slope.c_zx = ray.ox - slope.xbyz*ray.oz;
    slope.c_zy = ray.oy - slope.ybyz*ray.oz;

    return slope;
}

// min(min(a, b), c)
__device__ __inline__ int min_vmin (int a, int b, int c) {
    int mvm;
    asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// max(max(a, b), c)
__device__ __inline__ int max_vmax (int a, int b, int c) {
    int mvm;
    asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// max(min(a, b), c)
__device__ __inline__ int max_vmin (int a, int b, int c) {
    int mvm;
    asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}
// min(max(a, b), c)
__device__ __inline__ int min_vmax (int a, int b, int c) {
    int mvm;
    asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(mvm) : "r"(a), "r"(b), "r"(c));
    return mvm;
}

__device__ __inline__ float minf_vminf (float f1, float f2, float f3) {
    return __int_as_float(min_vmin(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __inline__ float maxf_vmaxf (float f1, float f2, float f3) {
    return __int_as_float(max_vmax(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __inline__ float minf_vmaxf (float f1, float f2, float f3) {
    return __int_as_float(min_vmax(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}
__device__ __inline__ float maxf_vminf (float f1, float f2, float f3) {
    return __int_as_float(max_vmin(__float_as_int(f1),
                                   __float_as_int(f2),
                                   __float_as_int(f3)));
}

__device__ int AABB_hit_Aila_Laine_Karras(const float3 invd, const float4 origin,
                                          const float4 AABBx,
                                          const float4 AABBy,
                                          const float4 AABBz)
{
    float bx, tx, by, ty, bz, tz;
    float tmin, tmax;
    unsigned int hits = 0;

    bx = (AABBx.x - origin.x) * invd.x ;
    tx = (AABBx.y - origin.x) * invd.x ;
    by = (AABBy.x - origin.y) * invd.y ;
    ty = (AABBy.y - origin.y) * invd.y ;
    bz = (AABBz.x - origin.z) * invd.z ;
    tz = (AABBz.y - origin.z) * invd.z ;
    tmin = maxf_vmaxf( fmin(bx, tx), fmin(by, ty), maxf_vminf(bz, tz, 0) );
    tmax = minf_vminf( fmax(bx, tx), fmax(by, ty), minf_vmaxf(bz, tz, origin.w) );
    hits += (int)(tmax >= tmin);

    bx = (AABBx.z - origin.x) * invd.x ;
    tx = (AABBx.w - origin.x) * invd.x ;
    by = (AABBy.z - origin.y) * invd.y ;
    ty = (AABBy.w - origin.y) * invd.y ;
    bz = (AABBz.z - origin.z) * invd.z ;
    tz = (AABBz.w - origin.z) * invd.z ;
    tmin = maxf_vmaxf( fmin(bx, tx), fmin(by, ty), maxf_vminf(bz, tz, 0) );
    tmax = minf_vminf( fmax(bx, tx), fmax(by, ty), minf_vmaxf(bz, tz, origin.w) );
    hits += (int)(tmax >= tmin);

    return hits;
}

__device__ int AABB_hit_williams(const float3 invd, const float4 origin,
                                 const float4 AABBx,
                                 const float4 AABBy,
                                 const float4 AABBz)
{
    float Lbx, Ltx, Lby, Lty, Lbz, Ltz;
    float Rbx, Rtx, Rby, Rty, Rbz, Rtz;
    float L_tmin, L_tmax, R_tmin, R_tmax;

    if (invd.x >= 0) {
        Lbx = (AABBx.x - origin.x) * invd.x;
        Ltx = (AABBx.y - origin.x) * invd.x;
        Rbx = (AABBx.z - origin.x) * invd.x;
        Rtx = (AABBx.w - origin.x) * invd.x;
    }
    else {
        Lbx = (AABBx.y - origin.x) * invd.x;
        Ltx = (AABBx.x - origin.x) * invd.x;
        Rbx = (AABBx.w - origin.x) * invd.x;
        Rtx = (AABBx.z - origin.x) * invd.x;
    }
    if (invd.y >= 0) {
        Lby = (AABBy.x - origin.y) * invd.y;
        Lty = (AABBy.y - origin.y) * invd.y;
        Rby = (AABBy.z - origin.y) * invd.y;
        Rty = (AABBy.w - origin.y) * invd.y;
    }
    else {
        Lby = (AABBy.y - origin.y) * invd.y;
        Lty = (AABBy.x - origin.y) * invd.y;
        Rby = (AABBy.w - origin.y) * invd.y;
        Rty = (AABBy.z - origin.y) * invd.y;
    }
    if (invd.z >= 0) {
        Lbz = (AABBz.x - origin.z) * invd.z;
        Ltz = (AABBz.y - origin.z) * invd.z;
        Rbz = (AABBz.z - origin.z) * invd.z;
        Rtz = (AABBz.w - origin.z) * invd.z;
    }
    else {
        Lbz = (AABBz.y - origin.z) * invd.z;
        Ltz = (AABBz.x - origin.z) * invd.z;
        Rbz = (AABBz.w - origin.z) * invd.z;
        Rtz = (AABBz.z - origin.z) * invd.z;
    }

    L_tmin = fmax( fmax(Lbx, Lby), fmax(Lbz, 0) );
    L_tmax = fmin( fmin(Ltx, Lty), fmin(Ltz, origin.w) );
    R_tmin = fmax( fmax(Rbx, Rby), fmax(Rbz, 0) );
    R_tmax = fmin( fmin(Rtx, Rty), fmin(Rtz, origin.w) );

    return (int)(L_tmax >= L_tmin) + (int)(R_tmax >= R_tmin);
}

__device__ int AABB_hit_williams_noif(const float3 invd, const float4 origin,
                                      const float4 AABBx,
                                      const float4 AABBy,
                                      const float4 AABBz)
{
    float Lbx, Ltx, Lby, Lty, Lbz, Ltz;
    float Rbx, Rtx, Rby, Rty, Rbz, Rtz;
    float L_tmin, L_tmax, R_tmin, R_tmax;

    Lbx = (AABBx.x - origin.x) * invd.x;
    Ltx = (AABBx.y - origin.x) * invd.x;
    Lby = (AABBy.x - origin.y) * invd.y;
    Lty = (AABBy.y - origin.y) * invd.y;
    Lbz = (AABBz.x - origin.z) * invd.z;
    Ltz = (AABBz.y - origin.z) * invd.z;
    Rbx = (AABBx.z - origin.x) * invd.x;
    Rtx = (AABBx.w - origin.x) * invd.x;
    Rby = (AABBy.z - origin.y) * invd.y;
    Rty = (AABBy.w - origin.y) * invd.y;
    Rbz = (AABBz.z - origin.z) * invd.z;
    Rtz = (AABBz.w - origin.z) * invd.z;

    L_tmin = fmax( fmax(fmin(Lbx, Ltx), fmin(Lby, Lty)), fmax(fmin(Lbz, Ltz), 0) );
    L_tmax = fmin( fmin(fmax(Lbx, Ltx), fmax(Lby, Lty)), fmin(fmax(Lbz, Ltz), origin.w) );
    R_tmin = fmax( fmax(fmin(Rbx, Rtx), fmin(Rby, Rty)), fmax(fmin(Rbz, Rtz), 0) );
    R_tmax = fmin( fmin(fmax(Rbx, Rtx), fmax(Rby, Rty)), fmin(fmax(Rbz, Rtz), origin.w) );

    return (int)(L_tmax >= L_tmin) + (int)(R_tmax >= R_tmin);
}

__host__ __device__ bool AABB_hit_eisemann(const Ray& ray,
                                           const RaySlope& slope,
                                           const float bx,
                                           const float by,
                                           const float bz,
                                           const float tx,
                                           const float ty,
                                           const float tz)
{
    float ox = ray.ox;
    float oy = ray.oy;
    float oz = ray.oz;

    float dx = ray.dx;
    float dy = ray.dy;
    float dz = ray.dz;

    float l = ray.length;

    float xbyy = slope.xbyy;
    float ybyx = slope.ybyx;
    float ybyz = slope.ybyz;
    float zbyy = slope.zbyy;
    float xbyz = slope.xbyz;
    float zbyx = slope.zbyx;
    float c_xy = slope.c_xy;
    float c_xz = slope.c_xz;
    float c_yx = slope.c_yx;
    float c_yz = slope.c_yz;
    float c_zx = slope.c_zx;
    float c_zy = slope.c_zy;

    switch(ray.dclass)
    {
    case MMM:

        if ((ox < bx) || (oy < by) || (oz < bz))
            return false; // AABB entirely in wrong octant wrt ray origin.

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false; // Past length of ray.

        else if ((ybyx * bx - ty + c_xy > 0.0f) ||
                 (xbyy * by - tx + c_yx > 0.0f) ||
                 (ybyz * bz - ty + c_zy > 0.0f) ||
                 (zbyy * by - tz + c_yz > 0.0f) ||
                 (zbyx * bx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PMM:

        if ((ox > tx) || (oy < by) || (oz < bz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false;
        else if ((ybyx * tx - ty + c_xy > 0.0f) ||
                 (xbyy * by - bx + c_yx < 0.0f) ||
                 (ybyz * bz - ty + c_zy > 0.0f) ||
                 (zbyy * by - tz + c_yz > 0.0f) ||
                 (zbyx * tx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - bx + c_zx < 0.0f))
            return false;

        return true;

    case MPM:

        if ((ox < bx) || (oy > ty) || (oz < bz))
            return false;

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false;
        else if ((ybyx * bx - by + c_xy < 0.0f) ||
                 (xbyy * ty - tx + c_yx > 0.0f) ||
                 (ybyz * bz - by + c_zy < 0.0f) ||
                 (zbyy * ty - tz + c_yz > 0.0f) ||
                 (zbyx * bx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PPM:

        if ((ox > tx) || (oy > ty) || (oz < bz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
            return false;
        else if ((ybyx * tx - by + c_xy < 0.0f) ||
                 (xbyy * ty - bx + c_yx < 0.0f) ||
                 (ybyz * bz - by + c_zy < 0.0f) ||
                 (zbyy * ty - tz + c_yz > 0.0f) ||
                 (zbyx * tx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - bx + c_zx < 0.0f))
            return false;

        return true;

    case MMP:

        if ((ox < bx) || (oy < by) || (oz > tz))
            return false;

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * bx - ty + c_xy > 0.0f) ||
                 (xbyy * by - tx + c_yx > 0.0f) ||
                 (ybyz * tz - ty + c_zy > 0.0f) ||
                 (zbyy * by - bz + c_yz < 0.0f) ||
                 (zbyx * bx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PMP:

        if ((ox > tx) || (oy < by) || (oz > tz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * tx - ty + c_xy > 0.0f) ||
                 (xbyy * by - bx + c_yx < 0.0f) ||
                 (ybyz * tz - ty + c_zy > 0.0f) ||
                 (zbyy * by - bz + c_yz < 0.0f) ||
                 (zbyx * tx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - bx + c_zx < 0.0f))
            return false;

        return true;

    case MPP:

        if ((ox < bx) || (oy > ty) || (oz > tz))
            return false;

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * bx - by + c_xy < 0.0f) ||
                 (xbyy * ty - tx + c_yx > 0.0f) ||
                 (ybyz * tz - by + c_zy < 0.0f) ||
                 (zbyy * ty - bz + c_yz < 0.0f) ||
                 (zbyx * bx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - tx + c_zx > 0.0f))
            return false;

        return true;

    case PPP:

        if ((ox > tx) || (oy > ty) || (oz > tz))
            return false;

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
            return false;
        else if ((ybyx * tx - by + c_xy < 0.0f) ||
                 (xbyy * ty - bx + c_yx < 0.0f) ||
                 (ybyz * tz - by + c_zy < 0.0f) ||
                 (zbyy * ty - bz + c_yz < 0.0f) ||
                 (zbyx * tx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - bx + c_zx < 0.0f))
            return false;

        return true;
    }

    return false;
}

__host__ __device__ bool AABB_hit_plucker(const Ray& ray,
                                          const float bx,
                                          const float by,
                                          const float bz,
                                          const float tx,
                                          const float ty,
                                          const float tz)
{
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;
    float length = ray.length;

    float s2bx, s2by, s2bz; // Vector from ray start to lower cell corner.
    float s2tx, s2ty, s2tz; // Vector from ray start to upper cell corner.

    s2bx = bx - ray.ox;
    s2by = by - ray.oy;
    s2bz = bz - ray.oz;

    s2tx = tx - ray.ox;
    s2ty = ty - ray.oy;
    s2tz = tz - ray.oz;

    switch(ray.dclass)
    {
        // MMM
        case 0:
        if (s2bx > 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2tx < 0.0f ||
            rx*s2ty - ry*s2bx > 0.0f ||
            rx*s2tz - rz*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx < 0.0f ||
            ry*s2bz - rz*s2ty < 0.0f ||
            ry*s2tz - rz*s2by > 0.0f ) return false;
        break;

        // PMM
        case 1:
        if (s2tx < 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2tx < 0.0f ||
            rx*s2by - ry*s2bx > 0.0f ||
            rx*s2bz - rz*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx < 0.0f ||
            ry*s2bz - rz*s2ty < 0.0f ||
            ry*s2tz - rz*s2by > 0.0f) return false;
        break;

        // MPM
        case 2:
        if (s2bx > 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2by - ry*length > 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2bx < 0.0f ||
            rx*s2ty - ry*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx < 0.0f ||
            ry*s2tz - rz*s2ty < 0.0f ||
            ry*s2bz - rz*s2by > 0.0f) return false;
        break;

        // PPM
        case 3:
        if (s2tx < 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2by - ry*length > 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2bx < 0.0f ||
            rx*s2by - ry*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx < 0.0f ||
            ry*s2tz - rz*s2ty < 0.0f ||
            ry*s2bz - rz*s2by > 0.0f) return false;
        break;

        // MMP
        case 4:
        if (s2bx > 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2tx < 0.0f ||
            rx*s2ty - ry*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx < 0.0f ||
            ry*s2bz - rz*s2by < 0.0f ||
            ry*s2tz - rz*s2ty > 0.0f) return false;
        break;

        // PMP
        case 5:
        if (s2tx < 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2tx < 0.0f ||
            rx*s2by - ry*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx < 0.0f ||
            ry*s2bz - rz*s2by < 0.0f ||
            ry*s2tz - rz*s2ty > 0.0f) return false;
        break;

        // MPP
        case 6:
        if (s2bx > 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2by - ry*length > 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2bx < 0.0f ||
            rx*s2ty - ry*s2tx > 0.0f ||
            rx*s2tz - rz*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx < 0.0f ||
            ry*s2tz - rz*s2by < 0.0f ||
            ry*s2bz - rz*s2ty > 0.0f) return false;
        break;

        // PPP
        case 7:
        if (s2tx < 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2by - ry*length > 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2bx < 0.0f ||
            rx*s2by - ry*s2tx > 0.0f ||
            rx*s2bz - rz*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx < 0.0f ||
            ry*s2tz - rz*s2by < 0.0f ||
            ry*s2bz - rz*s2ty > 0.0f) return false;
        break;
    }

    // Didn't return false above, so we have a hit.
    return true;

}

__global__ void AABB_hit_Aila_Laine_Karras_kernel(const Ray* rays,
                                           const float4* AABBs,
                                           const int N_rays,
                                           const int N_AABBs,
                                           unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz, origin;
    Ray ray;
    float3 invd;
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        hit_count = 0;

        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        origin.x = ray.ox * invd.x;
        origin.y = ray.oy * invd.y;
        origin.z = ray.oz * invd.z;
        origin.w = ray.length;

        for (int i=0; i<N_AABBs/2; i++)
        {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];
            hit_count += AABB_hit_Aila_Laine_Karras(invd, origin, AABBx, AABBy, AABBz);
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void AABB_hit_williams_kernel(const Ray* rays,
                                         const float4* AABBs,
                                         const int N_rays,
                                         const int N_AABBs,
                                         unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz, origin;
    Ray ray;
    float3 invd;
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        hit_count = 0;

        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        origin.x = ray.ox * invd.x;
        origin.y = ray.oy * invd.y;
        origin.z = ray.oz * invd.z;
        origin.w = ray.length;

        for (int i=0; i<N_AABBs/2; i++)
        {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];
            hit_count += AABB_hit_williams(invd, origin, AABBx, AABBy, AABBz);
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void AABB_hit_williams_noif_kernel(const Ray* rays,
                                              const float4* AABBs,
                                              const int N_rays,
                                              const int N_AABBs,
                                              unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz, origin;
    Ray ray;
    float3 invd;
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        hit_count = 0;

        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        origin.x = ray.ox * invd.x;
        origin.y = ray.oy * invd.y;
        origin.z = ray.oz * invd.z;
        origin.w = ray.length;

        for (int i=0; i<N_AABBs/2; i++)
        {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];

            hit_count += AABB_hit_williams_noif(invd, origin, AABBx, AABBy, AABBz);
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void AABB_hit_eisemann_kernel(const Ray* rays,
                                         const float4* AABBs,
                                         const int N_rays,
                                         const int N_AABBs,
                                         unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz;
    Ray ray;
    RaySlope slope;
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        slope = ray_slope(ray);
        hit_count = 0;

        for (int i=0; i<N_AABBs/2; i++) {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];
            if (AABB_hit_eisemann(ray, slope,
                                  AABBx.x, AABBy.x, AABBz.x,
                                  AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (AABB_hit_eisemann(ray, slope,
                                  AABBx.z, AABBy.z, AABBz.z,
                                  AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void AABB_hit_plucker_kernel(const Ray* rays,
                                        const float4* AABBs,
                                        const int N_rays,
                                        const int N_AABBs,
                                        unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz;
    Ray ray;
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        hit_count = 0;

        for (int i=0; i<N_AABBs/2; i++) {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];
            if (AABB_hit_plucker(ray,
                                 AABBx.x, AABBy.x, AABBz.x,
                                 AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (AABB_hit_plucker(ray,
                                 AABBx.z, AABBy.z, AABBz.z,
                                 AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}


int main(int argc, char* argv[]) {

    /* Initialize run parameters. */

    unsigned int N_rays = 100000;
    unsigned int N_AABBs = 2*500;

    if (argc > 1)
        N_rays = (unsigned int) std::strtol(argv[1], NULL, 10);
    if (argc > 2)
        N_AABBs = 2*(unsigned int) std::strtol(argv[2], NULL, 10);

    std::cout << "Testing " << N_rays << " rays against "
              << N_AABBs << " AABBs." << std::endl;
    std::cout << std::endl;


    /* Generate the rays, emitted from (0, 0, 0) in a random direction.
     * NB: *Not* uniform random on surface of sphere.
     */

    thrust::host_vector<Ray> h_rays(N_rays);
    thrust::host_vector<float4> h_AABBs(3*(N_AABBs/2));
    thrust::host_vector<unsigned int> h_keys(N_rays);

    grace::random_float_functor rng(-1.0f, 1.0f);
    for (int i=0; i<N_rays; i++) {
        float x, y, z, N;

        x = rng(3*i + 0);
        y = rng(3*i + 1);
        z = rng(3*i + 2);

        N = sqrt(x*x + y*y + z*z);

        h_rays[i].dx = x / N;
        h_rays[i].dy = y / N;
        h_rays[i].dz = z / N;

        h_rays[i].ox = h_rays[i].oy = h_rays[i].oz = 0;

        h_rays[i].length = N;

        h_rays[i].dclass = 0;
        if (x >= 0)
            h_rays[i].dclass += 1;
        if (y >= 0)
            h_rays[i].dclass += 2;
        if (z >= 0)
            h_rays[i].dclass += 4;

        // Floats must be in (0, 1) for morton_key().
        h_keys[i] = grace::morton_key((h_rays[i].dx+1)/2.f,
                                      (h_rays[i].dy+1)/2.f,
                                      (h_rays[i].dz+1)/2.f);
    }
    // Sort rays by Morton key.  This has a ~4--5x performance impact on Kepler
    // for the Plucker and Eisemann kernels.
    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    h_keys.clear();
    h_keys.shrink_to_fit();


    /* Generate the AABBs, with all points uniformly random in [-1, 1). */

    float bx, tx, by, ty, bz, tz;
    for (int i=0; i<N_AABBs/2; i++) {
        // ~ Left child AABB.
        bx = rng(3*N_rays + 12*i+0);
        ty = rng(3*N_rays + 12*i+1);
        by = rng(3*N_rays + 12*i+2);
        ty = rng(3*N_rays + 12*i+3);
        bz = rng(3*N_rays + 12*i+4);
        tz = rng(3*N_rays + 12*i+5);

        h_AABBs[3*i + 0].x = min(bx, tx);
        h_AABBs[3*i + 1].x = min(by, ty);
        h_AABBs[3*i + 2].x = min(bz, tz);

        h_AABBs[3*i + 0].y = max(bx, tx);
        h_AABBs[3*i + 1].y = max(by, ty);
        h_AABBs[3*i + 2].y = max(bz, tz);

        // ~ Right child AABB.
        bx = rng(3*N_rays + 12*i+6);
        ty = rng(3*N_rays + 12*i+7);
        by = rng(3*N_rays + 12*i+8);
        ty = rng(3*N_rays + 12*i+9);
        bz = rng(3*N_rays + 12*i+10);
        tz = rng(3*N_rays + 12*i+11);

        h_AABBs[3*i + 0].z = min(bx, tx);
        h_AABBs[3*i + 1].z = min(by, ty);
        h_AABBs[3*i + 2].z = min(bz, tz);

        h_AABBs[3*i + 0].w = max(bx, tx);
        h_AABBs[3*i + 1].w = max(by, ty);
        h_AABBs[3*i + 2].w = max(bz, tz);
    }

    thrust::device_vector<Ray> d_rays = h_rays;
    thrust::device_vector<float4> d_AABBs = h_AABBs;
    thrust::device_vector<unsigned int> d_ray_hits(N_rays);
    thrust::host_vector<unsigned int> h_ray_hits(N_rays);


    /* Profile Aila and Laine. */

    // On GPU only.
    cudaEvent_t start, stop;
    float elapsed;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_Aila_Laine_Karras_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_aila_laine_ray_hits = d_ray_hits;

    // Print results.
    std::cout << "Aila, Laine and Karras:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    /* Profile Williams. */

    // On GPU only.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_williams_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_williams_ray_hits = d_ray_hits;

    // Print results.
    std::cout << "Williams:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    /* Profile Williams with no ifs. */

    // On GPU only.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_williams_noif_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_williams_noif_ray_hits = d_ray_hits;

    // Print results.
    std::cout << "Williams (no if):" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    /* Profile Eisemann. */

    // On GPU.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_eisemann_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_eisemann_ray_hits = d_ray_hits;

    // On CPU.
    double t = (double)clock() / CLOCKS_PER_SEC;
    for (int i=0; i<N_rays; i++) {
        Ray ray = h_rays[i];
        RaySlope slope = ray_slope(ray);
        unsigned int hit_count = 0;

        for (int j=0; j<N_AABBs/2; j++) {
            float4 AABBx = h_AABBs[3*j + 0];
            float4 AABBy = h_AABBs[3*j + 1];
            float4 AABBz = h_AABBs[3*j + 2];
            if (AABB_hit_eisemann(ray, slope,
                                  AABBx.x, AABBy.x, AABBz.x,
                                  AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (AABB_hit_eisemann(ray, slope,
                                  AABBx.z, AABBy.z, AABBz.z,
                                  AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
            h_ray_hits[i] = hit_count;
        }
    }
    t = (double)clock() / CLOCKS_PER_SEC - t;

    // Print results.
    std::cout << "Eisemann:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    std::cout << "    CPU: " << t*1000. << " ms." << std::endl;


    /* Profile Plucker. */

    // On GPU.
    cudaEventRecord(start);
    AABB_hit_plucker_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_plucker_ray_hits = d_ray_hits;

    // On CPU.
    thrust::fill(h_ray_hits.begin(), h_ray_hits.end(), 0u);
    t = (double)clock() / CLOCKS_PER_SEC;
    for (int i=0; i<N_rays; i++) {
        Ray ray = h_rays[i];
        unsigned int hit_count = 0;

        for (int j=0; j<N_AABBs/2; j++) {
            float4 AABBx = h_AABBs[3*j + 0];
            float4 AABBy = h_AABBs[3*j + 1];
            float4 AABBz = h_AABBs[3*j + 2];
            if (AABB_hit_plucker(ray,
                                 AABBx.x, AABBy.x, AABBz.x,
                                 AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (AABB_hit_plucker(ray,
                                 AABBx.z, AABBy.z, AABBz.z,
                                 AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
            h_ray_hits[i] = hit_count;
        }
    }
    t = (double)clock() / CLOCKS_PER_SEC - t;

    // Print results.
    std::cout << "Plucker:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    std::cout << "    CPU: " << t*1000. << " ms." << std::endl;
    std::cout << std::endl;


    /* Check Plucker and Eisemann intersection results match. */

    for (int i=0; i<N_rays; i++)
    {
        if (h_eisemann_ray_hits[i] != h_plucker_ray_hits[i])
        {
            std::cout << "Ray " << i << ": Eisemann (" << h_eisemann_ray_hits[i]
                      << ") != (" << h_plucker_ray_hits[i] << ") Plucker!"
                      << std::endl;
            std::cout << "ray.dclass = " << h_rays[i].dclass << std::endl;
            std::cout << "ray (ox, oy, oz): (" << h_rays[i].ox << ", "
                      << h_rays[i].oy << ", " << h_rays[i].oz << ")."
                      << std::endl;
            std::cout << "ray (dx, dy, dz): (" << h_rays[i].dx << ", "
                      << h_rays[i].dy << ", " << h_rays[i].dz << ")."
                      << std::endl;
            std::cout << std::endl;
        }
    }

    /* Check Aila and Plucker intersection results match. */

    for (int i=0; i<N_rays; i++)
    {
        if (h_aila_laine_ray_hits[i] != h_plucker_ray_hits[i])
        {
            std::cout << "Ray " << i << ": Aila (" << h_aila_laine_ray_hits[i]
                      << ") != (" << h_plucker_ray_hits[i] << ") Plucker!"
                      << std::endl;
            std::cout << "ray.dclass = " << h_rays[i].dclass << std::endl;
            std::cout << "ray (ox, oy, oz): (" << h_rays[i].ox << ", "
                      << h_rays[i].oy << ", " << h_rays[i].oz << ")."
                      << std::endl;
            std::cout << "ray (dx, dy, dz): (" << h_rays[i].dx << ", "
                      << h_rays[i].dy << ", " << h_rays[i].dz << ")."
                      << std::endl;
            std::cout << std::endl;
        }
    }

    /* Check Williams and Plucker intersection results match. */

    for (int i=0; i<N_rays; i++)
    {
        if (h_williams_ray_hits[i] != h_plucker_ray_hits[i])
        {
            std::cout << "Ray " << i << ": Williams (" << h_williams_ray_hits[i]
                      << ") != (" << h_plucker_ray_hits[i] << ") Plucker!"
                      << std::endl;
            std::cout << "ray.dclass = " << h_rays[i].dclass << std::endl;
            std::cout << "ray (ox, oy, oz): (" << h_rays[i].ox << ", "
                      << h_rays[i].oy << ", " << h_rays[i].oz << ")."
                      << std::endl;
            std::cout << "ray (dx, dy, dz): (" << h_rays[i].dx << ", "
                      << h_rays[i].dy << ", " << h_rays[i].dz << ")."
                      << std::endl;
            std::cout << std::endl;
        }
    }

    /* Check Williams (no ifs) and Plucker intersection results match. */

    for (int i=0; i<N_rays; i++)
    {
        if (h_williams_noif_ray_hits[i] != h_plucker_ray_hits[i])
        {
            std::cout << "Ray " << i << ": Williams no ifs (" << h_williams_noif_ray_hits[i]
                      << ") != (" << h_plucker_ray_hits[i] << ") Plucker!"
                      << std::endl;
            std::cout << "ray.dclass = " << h_rays[i].dclass << std::endl;
            std::cout << "ray (ox, oy, oz): (" << h_rays[i].ox << ", "
                      << h_rays[i].oy << ", " << h_rays[i].oz << ")."
                      << std::endl;
            std::cout << "ray (dx, dy, dz): (" << h_rays[i].dx << ", "
                      << h_rays[i].dy << ", " << h_rays[i].dz << ")."
                      << std::endl;
            std::cout << std::endl;
        }
    }

    return 0;

}
