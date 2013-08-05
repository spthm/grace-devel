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

__device__ float3 uniform_sample_cone(float u1, float u2, float costhetamax,
                                      const float3 &x,
                                      const float3 &y,
                                      const float3 &z)
{
    float costheta = (1-u1)+u1*costhetamax,
          sintheta = sqrt(1-costheta*costheta),
          phi = u2*2*M_PIf;

    return cos(phi)*sintheta*x + sin(phi)*sintheta*y +  costheta*z;
}

__device__ float uniform_cone_pdf(float costhetamax)
{
    return 1.f / (2*M_PIf*(1-costhetamax));
}

__device__ void coordinate_system(const float3 &v1, float3 *v2, float3 *v3)
{
    if(fabs(v1.x) > fabs(v1.y))
    {
        float invLen = rsqrt(v1.x*v1.x+v1.z*v1.z);
        *v2 = make_float3(-v1.z*invLen, 0, v1.x*invLen);
    }
    else
    {
        float invLen = rsqrt(v1.y*v1.y+v1.z*v1.z);
        *v2 = make_float3(0, v1.z*invLen, -v1.y*invLen);
    }

    *v3 = cross(v1, *v2);
}

__device__ float3 sample_surface(const Sphere &s, 
                                 const float3 &p, float u1, float u2,
                                 float *pdf=NULL)
{
    float3 dv = s.center - p;

    float dist2 = dot(dv,dv);

    float3 wc = unit(dv),
           wcX, wcY;

    coordinate_system(wc, &wcX, &wcY);

    float sinThetaMax2 = s.radius*s.radius / dist2,
          cosThetaMax = sqrtf(max(0.f, 1.f - sinThetaMax2));

    float3 dir = uniform_sample_cone(u1, u2, cosThetaMax, wcX, wcY, wc);

    float thit;
    if(!intersect(p, dir, s, &thit))
        thit = dot(dv, dir);

    if(pdf)
        *pdf = uniform_cone_pdf(cosThetaMax);

    return p + thit*dir;
}
