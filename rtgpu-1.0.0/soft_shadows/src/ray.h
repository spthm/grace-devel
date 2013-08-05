#ifndef CUDA_RAY_H
#define CUDA_RAY_H

inline __device__ bool intersects(
    const float3 &ray_ori, const float3 &inv_ray_dir,
    float mint, float maxt, const AABB &aabb, float *t_hit=NULL)
{
    float3 aabb_lower = aabb.center - aabb.hsize,
           aabb_upper = aabb.center + aabb.hsize;

    float t0 = mint, t1 = maxt;

    // test X slab
    float tnear = (aabb_lower.x - ray_ori.x)*inv_ray_dir.x,
          tfar = (aabb_upper.x - ray_ori.x)*inv_ray_dir.x;
    if(tnear > tfar)
        swap(tnear, tfar);
    t0 = max(tnear,t0);
    t1 = min(tfar,t1);
    if(t0 <= t1)
    {
        // test Y slab
        tnear = (aabb_lower.y - ray_ori.y)*inv_ray_dir.y;
        tfar = (aabb_upper.y - ray_ori.y)*inv_ray_dir.y;
        if(tnear > tfar)
            swap(tnear, tfar);
        t0 = max(tnear,t0);
        t1 = min(tfar,t1);
        if(t0 <= t1)
        {
            // test Z slab
            tnear = (aabb_lower.z - ray_ori.z)*inv_ray_dir.z;
            tfar = (aabb_upper.z - ray_ori.z)*inv_ray_dir.z;
            if(tnear > tfar)
                swap(tnear, tfar);
            t0 = max(tnear,t0);
            t1 = min(tfar,t1);
            if(t0 <= t1)
            {
                if(t_hit)
                    *t_hit = t1;
                return true;
            }
        }
    }
    return false;
}

#endif
