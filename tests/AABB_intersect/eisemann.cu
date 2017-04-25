#include "eisemann.cuh"
#include "auxillary.cuh"

__host__ __device__ EisemannRayAuxillary eisemann_auxillary(const Ray& ray)
{
    EisemannRayAuxillary aux;

    set_dclass(ray, aux);
    set_slope(ray, aux);

    return aux;
}

// NOTE: This is actually an incomplete implementation, because the ray slopes
// method requires special handling any time a ray has a direction component
// exactly equal to zero.

__host__ __device__ int eisemann(const Ray& ray,
                                 const EisemannRayAuxillary& aux,
                                 const AABB& box)
{
    float ox = ray.ox;
    float oy = ray.oy;
    float oz = ray.oz;

    float dx = ray.dx;
    float dy = ray.dy;
    float dz = ray.dz;

    // Assume start == 0.
    float l = ray.end;

    float xbyy = aux.xbyy;
    float ybyx = aux.ybyx;
    float ybyz = aux.ybyz;
    float zbyy = aux.zbyy;
    float xbyz = aux.xbyz;
    float zbyx = aux.zbyx;
    float c_xy = aux.c_xy;
    float c_xz = aux.c_xz;
    float c_yx = aux.c_yx;
    float c_yz = aux.c_yz;
    float c_zx = aux.c_zx;
    float c_zy = aux.c_zy;

    float bx = box.bx;
    float by = box.by;
    float bz = box.bz;
    float tx = box.tx;
    float ty = box.ty;
    float tz = box.tz;

    switch(aux.dclass) {

    case MMM:

        if ((ox < bx) || (oy < by) || (oz < bz))
        {
            // AABB entirely in wrong octant wrt ray origin.
            return MISS;
        }

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
        {
            // Past length of ray.
            return MISS;
        }


        else if ((ybyx * bx - ty + c_xy > 0.0f) ||
                 (xbyy * by - tx + c_yx > 0.0f) ||
                 (ybyz * bz - ty + c_zy > 0.0f) ||
                 (zbyy * by - tz + c_yz > 0.0f) ||
                 (zbyx * bx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - tx + c_zx > 0.0f))
        {
            return MISS;
        }

        // Didn't return miss above, so we have a hit.
        return HIT;

    case PMM:

        if ((ox > tx) || (oy < by) || (oz < bz))
        {
            return MISS;
        }

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 tz - oz - dz*l < 0.0f)
        {
            return MISS;
        }

        else if ((ybyx * tx - ty + c_xy > 0.0f) ||
                 (xbyy * by - bx + c_yx < 0.0f) ||
                 (ybyz * bz - ty + c_zy > 0.0f) ||
                 (zbyy * by - tz + c_yz > 0.0f) ||
                 (zbyx * tx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - bx + c_zx < 0.0f))
        {
            return MISS;
        }

        return HIT;

    case MPM:

        if ((ox < bx) || (oy > ty) || (oz < bz))
        {
            return MISS;
        }

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
        {
            return MISS;
        }

        else if ((ybyx * bx - by + c_xy < 0.0f) ||
                 (xbyy * ty - tx + c_yx > 0.0f) ||
                 (ybyz * bz - by + c_zy < 0.0f) ||
                 (zbyy * ty - tz + c_yz > 0.0f) ||
                 (zbyx * bx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - tx + c_zx > 0.0f))
        {
            return MISS;
        }

        return HIT;

    case PPM:

        if ((ox > tx) || (oy > ty) || (oz < bz))
        {
            return MISS;
        }

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 tz - oz - dz*l < 0.0f)
        {
            return MISS;
        }

        else if ((ybyx * tx - by + c_xy < 0.0f) ||
                 (xbyy * ty - bx + c_yx < 0.0f) ||
                 (ybyz * bz - by + c_zy < 0.0f) ||
                 (zbyy * ty - tz + c_yz > 0.0f) ||
                 (zbyx * tx - tz + c_xz > 0.0f) ||
                 (xbyz * bz - bx + c_zx < 0.0f))
        {
            return MISS;
        }

        return HIT;

    case MMP:

        if ((ox < bx) || (oy < by) || (oz > tz))
        {
            return MISS;
        }

        else if (tx - ox - dx*l < 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
        {
            return MISS;
        }

        else if ((ybyx * bx - ty + c_xy > 0.0f) ||
                 (xbyy * by - tx + c_yx > 0.0f) ||
                 (ybyz * tz - ty + c_zy > 0.0f) ||
                 (zbyy * by - bz + c_yz < 0.0f) ||
                 (zbyx * bx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - tx + c_zx > 0.0f))
        {
            return MISS;
        }

        return HIT;

    case PMP:

        if ((ox > tx) || (oy < by) || (oz > tz))
        {
            return MISS;
        }

        else if (bx - ox - dx*l > 0.0f ||
                 ty - oy - dy*l < 0.0f ||
                 bz - oz - dz*l > 0.0f)
        {
            return MISS;
        }

        else if ((ybyx * tx - ty + c_xy > 0.0f) ||
                 (xbyy * by - bx + c_yx < 0.0f) ||
                 (ybyz * tz - ty + c_zy > 0.0f) ||
                 (zbyy * by - bz + c_yz < 0.0f) ||
                 (zbyx * tx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - bx + c_zx < 0.0f))
        {
            return MISS;
        }

        return HIT;

    case MPP:

        if ((ox < bx) || (oy > ty) || (oz > tz))
        {
            return MISS;
        }

        else if (tx - ox - dx*l < 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
        {
            return MISS;
        }

        else if ((ybyx * bx - by + c_xy < 0.0f) ||
                 (xbyy * ty - tx + c_yx > 0.0f) ||
                 (ybyz * tz - by + c_zy < 0.0f) ||
                 (zbyy * ty - bz + c_yz < 0.0f) ||
                 (zbyx * bx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - tx + c_zx > 0.0f))
        {
            return MISS;
        }

        return HIT;

    case PPP:

        if ((ox > tx) || (oy > ty) || (oz > tz))
        {
            return MISS;
        }

        else if (bx - ox - dx*l > 0.0f ||
                 by - oy - dy*l > 0.0f ||
                 bz - oz - dz*l > 0.0f)
        {
            return MISS;
        }

        else if ((ybyx * tx - by + c_xy < 0.0f) ||
                 (xbyy * ty - bx + c_yx < 0.0f) ||
                 (ybyz * tz - by + c_zy < 0.0f) ||
                 (zbyy * ty - bz + c_yz < 0.0f) ||
                 (zbyx * tx - bz + c_xz < 0.0f) ||
                 (xbyz * tz - bx + c_zx < 0.0f))
        {
            return MISS;
        }

        return HIT;
    }

    // Ray class unknown, better return a miss.
    return MISS;
}
