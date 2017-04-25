#include "aila_laine_karras.cuh"
#include "auxillary.cuh"

#include "grace/cuda/detail/device/intrinsics.cuh"

#include <math.h>

__host__ __device__ AilaRayAuxillary aila_auxillary(const Ray& ray)
{
    AilaRayAuxillary aux;

    set_invd(ray, aux);

    return aux;
}

__device__ int aila(const Ray& ray, const AilaRayAuxillary& aux,
                    const AABB& box)
{
    using namespace grace; // for maxf_vminf etc.

    float bx = box.bx;
    float by = box.by;
    float bz = box.bz;
    float tx = box.tx;
    float ty = box.ty;
    float tz = box.tz;

    bx = (bx - ray.ox) * aux.invdx;
    tx = (tx - ray.ox) * aux.invdx;
    by = (by - ray.oy) * aux.invdy;
    ty = (ty - ray.oy) * aux.invdy;
    bz = (bz - ray.oz) * aux.invdz;
    tz = (tz - ray.oz) * aux.invdz;

    float tmin, tmax;
    // Assume start == 0.
    tmin = maxf_vmaxf( fmin(bx, tx), fmin(by, ty), maxf_vminf(bz, tz, 0) );
    tmax = minf_vminf( fmax(bx, tx), fmax(by, ty), minf_vmaxf(bz, tz, ray.end) );

    return (tmax >= tmin ? HIT : MISS);
}
