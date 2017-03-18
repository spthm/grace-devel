#include "triangle.cuh"

__host__ __device__ void TriangleAABB::operator()(
    const Triangle& tri, grace::AABB<float>* aabb) const
{
    grace::Vector<3, float> v0 = tri.v;
    grace::Vector<3, float> v1 = v0 + tri.e1;
    grace::Vector<3, float> v2 = v0 + tri.e2;

    aabb->min.x = min(v0.x, min(v1.x, v2.x));
    aabb->min.y = min(v0.y, min(v1.y, v2.y));
    aabb->min.z = min(v0.z, min(v1.z, v2.z));

    aabb->max.x = max(v0.x, max(v1.x, v2.x));
    aabb->max.y = max(v0.y, max(v1.y, v2.y));
    aabb->max.z = max(v0.z, max(v1.z, v2.z));

// Some PLY files contain triangles which are zero-sized in one or more
// dimensions. In debug mode, GRACE has an assertion check that lower x/y/z
// bounds are _less_ than x/y/z upper bounds. The below is therefore needed to
// prevent run-time errors in debug mode; however, it should not be allowed to
// impact performance when not in debug mode.
#ifdef GRACE_DEBUG
    if (aabb->min.x == aabb->max.x) {
        float scale = abs(aabb->min.x);
        aabb->min.x -= AABB_EPSILON * scale;
        aabb->max.x += AABB_EPSILON * scale;
    }
    if (aabb->min.y == aabb->max.y) {
        float scale = abs(aabb->min.y);
        aabb->min.y -= AABB_EPSILON * scale;
        aabb->max.y += AABB_EPSILON * scale;
    }
    if (aabb->min.z == aabb->max.z) {
        float scale = abs(aabb->min.z);
        aabb->min.z -= AABB_EPSILON * scale;
        aabb->max.z += AABB_EPSILON * scale;
    }
#endif
}
