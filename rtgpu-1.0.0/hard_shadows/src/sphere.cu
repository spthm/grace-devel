#include "types.h"

__device__ bool intersect(const float3 &ray_ori, const float3 &ray_dir, 
                          const Sphere &sphere, 
                          float *t_ray, float3 *normal=NULL)
{
    float3 O = ray_ori - sphere.center;
    float3 D = ray_dir;

    float b = dot(O, D);
    float c = dot(O, O)-sphere.radius*sphere.radius;
    float d = b*b-c;

    // the ray's line cross cross the sphere at 2 points
    if(d > 0.0f)
    {
	d = sqrtf(d);
	float t0 = (-b - d);


#if ROBUST
	float dt0 = 0.0f;
	bool do_refine = false;
	if(fabsf(t0) > 10.f * sphere.radius) 
	    do_refine = true;

	if(do_refine) 
	{
	    // refine t1
	    float3 O1 = O + t0 * D;
	    b = dot(O1, D);
	    c = dot(O1, O1) - sphere.radius*sphere.radius;
	    d = b*b - c;

	    if(d > 0.0f) 
	    {
                d = sqrtf(d);
		dt0 = (-b - d);
                t0 += dt0;
	    }
	}
#endif

        // the ray's origin is outside the sphere?
	if(t0 > RAY_EPSILON)
	{
            if(t0 < *t_ray)
            {
                if(normal)
                    *normal = (O + t0*D)/sphere.radius;

                if(t_ray)
                    *t_ray = t0;

                return true;
            }
            else
                return false;
	} 

        float t1 = (-b + d);
#if ROBUST
        t1 += (do_refine ? dt0 : 0f);
#endif
        if(t1 > RAY_EPSILON) 
        {
            if(t1 < *t_ray)
            {
                if(normal)
                    *normal = (O + t1*D)/sphere.radius;

                if(t_ray)
                    *t_ray = t1;

                return true;
            }
            else
                return false;
        }
    }

    return false;
}

__device__ void bounds(const Sphere &sphere, float3 *lower, float3 *upper)
{
    *lower = sphere.center - sphere.radius;
    *upper = sphere.center + sphere.radius;
}
