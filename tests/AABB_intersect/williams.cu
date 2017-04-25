#include "williams.cuh"
#include "auxillary.cuh"

__host__ __device__ WilliamsRayAuxillary williams_auxillary(const Ray& ray)
{
    WilliamsRayAuxillary aux;

    set_invd(ray, aux);

    return aux;
}

__host__ __device__ int williams(const Ray& ray,
                                 const WilliamsRayAuxillary& aux,
                                 const AABB& box)
{
    float bx, by, bz;
    float tx, ty, tz;

    if (aux.invdx >= 0) {
        bx = (box.bx - ray.ox) * aux.invdx;
        tx = (box.tx - ray.ox) * aux.invdx;
    }
    else {
        bx = (box.tx - ray.ox) * aux.invdx;
        tx = (box.bx - ray.ox) * aux.invdx;
    }
    if (aux.invdy >= 0) {
        by = (box.by - ray.oy) * aux.invdy;
        ty = (box.ty - ray.oy) * aux.invdy;
    }
    else {
        by = (box.ty - ray.oy) * aux.invdy;
        ty = (box.by - ray.oy) * aux.invdy;
    }
    if (aux.invdz >= 0) {
        bz = (box.bz - ray.oz) * aux.invdz;
        tz = (box.tz - ray.oz) * aux.invdz;
    }
    else {
        bz = (box.tz - ray.oz) * aux.invdz;
        tz = (box.bz - ray.oz) * aux.invdz;
    }

    float tmin, tmax;
    // Assume start == 0.
    tmin = fmax( fmax(bx, by), fmax(bz, 0) );
    tmax = fmin( fmin(tx, ty), fmin(tz, ray.end) );

    return (tmax >= tmin ? HIT : MISS);
}

__host__ __device__ int williams_noif(const Ray& ray,
                                      const WilliamsRayAuxillary& aux,
                                      const AABB& box)
{
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
    tmin = fmax( fmax(fmin(bx, tx), fmin(by, ty)), fmax(fmin(bz, tz), 0) );
    tmax = fmin( fmin(fmax(bx, tx), fmax(by, ty)), fmin(fmax(bz, tz), ray.end) );

    return (tmax >= tmin ? HIT : MISS);
}
