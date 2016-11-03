#include "triangle.cuh"

__host__ __device__ void TriangleAABB::operator()(
    const Triangle& tri, float3* bot, float3* top) const
{
    float3 v0 = tri.v;
    float3 v1 = v0 + tri.e1;
    float3 v2 = v0 + tri.e2;

    bot->x = min(v0.x, min(v1.x, v2.x));
    bot->y = min(v0.y, min(v1.y, v2.y));
    bot->z = min(v0.z, min(v1.z, v2.z));

    top->x = max(v0.x, max(v1.x, v2.x));
    top->y = max(v0.y, max(v1.y, v2.y));
    top->z = max(v0.z, max(v1.z, v2.z));

    // Some PLY files contain triangles which are zero-sized in one or more
    // dimensions, but GRACE is not robust to zero-sized AABBs.
    if (bot->x == top->x) {
        float scale = abs(bot->x);
        bot->x -= AABB_EPSILON * scale;
        top->x += AABB_EPSILON * scale;
    }
    if (bot->y == top->y) {
        float scale = abs(bot->y);
        bot->y -= AABB_EPSILON * scale;
        top->y += AABB_EPSILON * scale;
    }
    if (bot->z == top->z) {
        float scale = abs(bot->z);
        bot->z -= AABB_EPSILON * scale;
        top->z += AABB_EPSILON * scale;
    }
}
