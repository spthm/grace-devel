#include "williams.cuh"

__host__ __device__ int williams(const Ray& ray, const AABB& box)
{
    float bx, by, bz;
    float tx, ty, tz;

    if (ray.invdx >= 0) {
        bx = (box.bx - ray.ox) * ray.invdx;
        tx = (box.tx - ray.ox) * ray.invdx;
    }
    else {
        bx = (box.tx - ray.ox) * ray.invdx;
        tx = (box.bx - ray.ox) * ray.invdx;
    }
    if (ray.invdy >= 0) {
        by = (box.by - ray.oy) * ray.invdy;
        ty = (box.ty - ray.oy) * ray.invdy;
    }
    else {
        by = (box.ty - ray.oy) * ray.invdy;
        ty = (box.by - ray.oy) * ray.invdy;
    }
    if (ray.invdz >= 0) {
        bz = (box.bz - ray.oz) * ray.invdz;
        tz = (box.tz - ray.oz) * ray.invdz;
    }
    else {
        bz = (box.tz - ray.oz) * ray.invdz;
        tz = (box.bz - ray.oz) * ray.invdz;
    }

    float tmin, tmax;
    tmin = fmax( fmax(bx, by), fmax(bz, 0) );
    tmax = fmin( fmin(tx, ty), fmin(tz, ray.length) );

    return (tmax >= tmin ? HIT : MISS);
}

__host__ __device__ int williams_noif(const Ray& ray, const AABB& box)
{
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
    tmin = fmax( fmax(fmin(bx, tx), fmin(by, ty)), fmax(fmin(bz, tz), 0) );
    tmax = fmin( fmin(fmax(bx, tx), fmax(by, ty)), fmin(fmax(bz, tz), ray.length) );

    return (tmax >= tmin ? HIT : MISS);
}
