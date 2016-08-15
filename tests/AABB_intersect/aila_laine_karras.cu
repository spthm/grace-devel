#include "aila_laine_karras.cuh"

#include "grace/cuda/detail/device/intrinsics.cuh"

#include <math.h>

__device__ int aila_laine_karras(const Ray& ray, const AABB& box)
{
    using namespace grace; // for maxf_vminf etc.

    float bx = box.bx;
    float by = box.by;
    float bz = box.bz;
    float tx = box.tx;
    float ty = box.ty;
    float tz = box.tz;

    bx = (bx - ray.ox) * ray.invdx;
    tx = (tx - ray.ox) * ray.invdx;
    by = (by - ray.oy) * ray.invdy;
    ty = (ty - ray.oy) * ray.invdy;
    bz = (bz - ray.oz) * ray.invdz;
    tz = (tz - ray.oz) * ray.invdz;

    float tmin, tmax;
    tmin = maxf_vmaxf( fmin(bx, tx), fmin(by, ty), maxf_vminf(bz, tz, 0) );
    tmax = minf_vminf( fmax(bx, tx), fmax(by, ty), minf_vmaxf(bz, tz, ray.length) );

    return (tmax >= tmin ? HIT : MISS);
}
