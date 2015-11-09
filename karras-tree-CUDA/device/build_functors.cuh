#pragma once

#include "../error.h"
#include "../types.h"

namespace grace {

//-----------------------------------------------------------------------------
// GPU helper functions for tree build kernels.
//-----------------------------------------------------------------------------

struct Delta_XOR
{
    GRACE_HOST_DEVICE uinteger32 operator()(
        const int i,
        const uinteger32* morton_keys,
        const size_t n_keys) const
    {
        // delta(-1) and delta(N-1) must return e.g. UINT_MAX because they
        // cover invalid ranges but are valid queries during tree construction.
        if (i < 0 || i + 1 >= n_keys)
            return uinteger32(-1);

        uinteger32 ki = morton_keys[i];
        uinteger32 kj = morton_keys[i+1];

        return ki ^ kj;
    }

    GRACE_HOST_DEVICE uinteger64 operator()(
        const int i,
        const uinteger64* morton_keys,
        const size_t n_keys) const
    {
        if (i < 0 || i + 1 >= n_keys)
            return uinteger64(-1);

        uinteger64 ki = morton_keys[i];
        uinteger64 kj = morton_keys[i+1];

        return ki ^ kj;

    }
};

// Euclidian distance metric.
struct Delta_sphere_euclidean
{
    GRACE_DEVICE float operator()(
        const int i,
        const float4* spheres,
        const size_t n_spheres) const
    {
        if (i < 0 || i + 1 >= n_spheres)
            return CUDART_INF_F;

        float4 si = spheres[i];
        float4 sj = spheres[i+1];

        return (si.x - sj.x) * (si.x - sj.x)
               + (si.y - sj.y) * (si.y - sj.y)
               + (si.z - sj.z) * (si.z - sj.z);
    }

    GRACE_DEVICE double operator()(
        const int i,
        const double4* spheres,
        const size_t n_spheres) const
    {
        if (i < 0 || i + 1 >= n_spheres)
            return CUDART_INF;

        double4 si = spheres[i];
        double4 sj = spheres[i+1];

        return (si.x - sj.x) * (si.x - sj.x)
               + (si.y - sj.y) * (si.y - sj.y)
               + (si.z - sj.z) * (si.z - sj.z);
    }
};

// Surface area 'distance' metric.
struct Delta_sphere_SA
{
    GRACE_DEVICE float operator()(
        const int i,
        const float4* spheres,
        const size_t n_spheres) const
    {
        if (i < 0 || i + 1 >= n_spheres)
            return CUDART_INF_F;

        float4 si = spheres[i];
        float4 sj = spheres[i+1];

        float L_x = max(si.x + si.w, sj.x + sj.w) - min(si.x - si.w, sj.x - sj.w);
        float L_y = max(si.y + si.w, sj.y + sj.w) - min(si.y - si.w, sj.y - sj.w);
        float L_z = max(si.z + si.w, sj.z + sj.w) - min(si.z - si.w, sj.z - sj.w);

        float SA = (L_x * L_y) + (L_x * L_z) + (L_y * L_z);

        return SA;
    }

    GRACE_DEVICE float operator()(
        const int i,
        const double4* spheres,
        const size_t n_spheres) const
    {
        if (i < 0 || i + 1 >= n_spheres)
            return CUDART_INF_F;

        double4 si = spheres[i];
        double4 sj = spheres[i+1];

        double L_x = max(si.x + si.w, sj.x + sj.w) - min(si.x - si.w, sj.x - sj.w);
        double L_y = max(si.y + si.w, sj.y + sj.w) - min(si.y - si.w, sj.y - sj.w);
        double L_z = max(si.z + si.w, sj.z + sj.w) - min(si.z - si.w, sj.z - sj.w);

        double SA = (L_x * L_y) + (L_x * L_z) + (L_y * L_z);

        return SA;
    }
};

struct AABB_sphere
{
    template <typename Real4>
    GRACE_HOST_DEVICE void operator()(
        Real4 sphere,
        float3* bot,
        float3* top) const
    {
        bot->x = sphere.x - sphere.w;
        top->x = sphere.x + sphere.w;

        bot->y = sphere.y - sphere.w;
        top->y = sphere.y + sphere.w;

        bot->z = sphere.z - sphere.w;
        top->z = sphere.z + sphere.w;
    }
};

} // namespace grace
