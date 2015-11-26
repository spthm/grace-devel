#include "plucker.cuh"

__host__ __device__ int plucker(const Ray& ray, const AABB& box)
{
    float dx = ray.dx;
    float dy = ray.dy;
    float dz = ray.dz;
    float l = ray.length;

    float bx = box.bx;
    float by = box.by;
    float bz = box.bz;
    float tx = box.tx;
    float ty = box.ty;
    float tz = box.tz;

    float o2bx, o2by, o2bz; // Vector from ray start to lower cell corner.
    float o2tx, o2ty, o2tz; // Vector from ray start to upper cell corner.

    o2bx = bx - ray.ox;
    o2by = by - ray.oy;
    o2bz = bz - ray.oz;

    o2tx = tx - ray.ox;
    o2ty = ty - ray.oy;
    o2tz = tz - ray.oz;

    switch(ray.dclass) {

    case MMM:

        if (o2bx > 0.0f || o2by > 0.0f || o2bz > 0.0f)
        {
            // AABB entirely in wrong octant wrt ray origin.
            return MISS;
        }

        if (o2tx - dx*l < 0.0f ||
            o2ty - dy*l < 0.0f ||
            o2tz - dz*l < 0.0f)
        {
            // Past length of ray.
            return MISS;
        }

        if (dx*o2by - dy*o2tx < 0.0f ||
            dx*o2ty - dy*o2bx > 0.0f ||
            dx*o2tz - dz*o2bx > 0.0f ||
            dx*o2bz - dz*o2tx < 0.0f ||
            dy*o2bz - dz*o2ty < 0.0f ||
            dy*o2tz - dz*o2by > 0.0f)
        {
            return MISS;
        }

        // Didn't return miss above, so we have a hit.
        return HIT;

    case PMM:

        if (o2tx < 0.0f || o2by > 0.0f || o2bz > 0.0f)
        {
            return MISS;
        }

        if (o2bx - dx*l > 0.0f ||
            o2ty - dy*l < 0.0f ||
            o2tz - dz*l < 0.0f)
        {
            return MISS;
        }

        if (dx*o2ty - dy*o2tx < 0.0f ||
            dx*o2by - dy*o2bx > 0.0f ||
            dx*o2bz - dz*o2bx > 0.0f ||
            dx*o2tz - dz*o2tx < 0.0f ||
            dy*o2bz - dz*o2ty < 0.0f ||
            dy*o2tz - dz*o2by > 0.0f)
        {
            return MISS;
        }

        return HIT;

    case MPM:

        if (o2bx > 0.0f || o2ty < 0.0f || o2bz > 0.0f)
        {
            return MISS;
        }

        if (o2tx - dx*l < 0.0f ||
            o2by - dy*l > 0.0f ||
            o2tz - dz*l < 0.0f)
        {
            return MISS;
        }

        if (dx*o2by - dy*o2bx < 0.0f ||
            dx*o2ty - dy*o2tx > 0.0f ||
            dx*o2tz - dz*o2bx > 0.0f ||
            dx*o2bz - dz*o2tx < 0.0f ||
            dy*o2tz - dz*o2ty < 0.0f ||
            dy*o2bz - dz*o2by > 0.0f)
        {
            return MISS;
        }

        return HIT;

    case PPM:

        if (o2tx < 0.0f || o2ty < 0.0f || o2bz > 0.0f)
        {
            return MISS;
        }

        if (o2bx - dx*l > 0.0f ||
            o2by - dy*l > 0.0f ||
            o2tz - dz*l < 0.0f)
        {
            return MISS;
        }

        if (dx*o2ty - dy*o2bx < 0.0f ||
            dx*o2by - dy*o2tx > 0.0f ||
            dx*o2bz - dz*o2bx > 0.0f ||
            dx*o2tz - dz*o2tx < 0.0f ||
            dy*o2tz - dz*o2ty < 0.0f ||
            dy*o2bz - dz*o2by > 0.0f)
        {
            return MISS;
        }

        return HIT;

    case MMP:

        if (o2bx > 0.0f || o2by > 0.0f || o2tz < 0.0f)
        {
            return MISS;
        }

        if (o2tx - dx*l < 0.0f ||
            o2ty - dy*l < 0.0f ||
            o2bz - dz*l > 0.0f)
        {
            return MISS;
        }

        if (dx*o2by - dy*o2tx < 0.0f ||
            dx*o2ty - dy*o2bx > 0.0f ||
            dx*o2tz - dz*o2tx > 0.0f ||
            dx*o2bz - dz*o2bx < 0.0f ||
            dy*o2bz - dz*o2by < 0.0f ||
            dy*o2tz - dz*o2ty > 0.0f)
        {
            return MISS;
        }

        return HIT;

    case PMP:

        if (o2tx < 0.0f || o2by > 0.0f || o2tz < 0.0f)
        {
            return MISS;
        }

        if (o2bx - dx*l > 0.0f ||
            o2ty - dy*l < 0.0f ||
            o2bz - dz*l > 0.0f)
        {
            return MISS;
        }

        if (dx*o2ty - dy*o2tx < 0.0f ||
            dx*o2by - dy*o2bx > 0.0f ||
            dx*o2bz - dz*o2tx > 0.0f ||
            dx*o2tz - dz*o2bx < 0.0f ||
            dy*o2bz - dz*o2by < 0.0f ||
            dy*o2tz - dz*o2ty > 0.0f)
        {
            return MISS;
        }

        return HIT;

    case MPP:

        if (o2bx > 0.0f || o2ty < 0.0f || o2tz < 0.0f)
        {
            return MISS;
        }

        if (o2tx - dx*l < 0.0f ||
            o2by - dy*l > 0.0f ||
            o2bz - dz*l > 0.0f)
        {
            return MISS;
        }

        if (dx*o2by - dy*o2bx < 0.0f ||
            dx*o2ty - dy*o2tx > 0.0f ||
            dx*o2tz - dz*o2tx > 0.0f ||
            dx*o2bz - dz*o2bx < 0.0f ||
            dy*o2tz - dz*o2by < 0.0f ||
            dy*o2bz - dz*o2ty > 0.0f)
        {
            return MISS;
        }

        return HIT;

    case PPP:

        if (o2tx < 0.0f || o2ty < 0.0f || o2tz < 0.0f)
        {
            return MISS;
        }

        if (o2bx - dx*l > 0.0f ||
            o2by - dy*l > 0.0f ||
            o2bz - dz*l > 0.0f)
        {
            return MISS;
        }

        if (dx*o2ty - dy*o2bx < 0.0f ||
            dx*o2by - dy*o2tx > 0.0f ||
            dx*o2bz - dz*o2tx > 0.0f ||
            dx*o2tz - dz*o2bx < 0.0f ||
            dy*o2tz - dz*o2by < 0.0f ||
            dy*o2bz - dz*o2ty > 0.0f)
        {
            return MISS;
        }

        return HIT;
    }

    // Ray class unknown, better return a miss.
    return MISS;
}
