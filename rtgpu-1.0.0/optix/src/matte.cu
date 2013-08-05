#include <optix.h>
#include "payload.h"
#include "light.h"
#include <curand_kernel.h>

#define NO_SHADOWS 0
#define SOFT_SHADOWS 1
#define HARD_SHADOWS 2

#define MODE SOFT_SHADOWS

#define SHADOW_STRATA 16
#define SQRT_SHADOW_STRATA 4
#define SHADOW_SAMPLES_PER_STRATUM 1
#define SHADOW_TOTAL_SAMPLES (SHADOW_STRATA*SHADOW_SAMPLES_PER_STRATUM)

rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);
rtDeclareVariable(float, t_hit, rtIntersectionDistance,);
rtDeclareVariable(float3, shading_normal, attribute shading_normal,);
rtDeclareVariable(RayPayload_radiance, payload_radiance, rtPayload,);
rtBuffer<shade::point_light> point_lights;
rtDeclareVariable(float, scene_epsilon,,);
rtDeclareVariable(rtObject, root_object,,);

struct sphere
{
    float3 center;
    float radius;
};

__device__ bool intersect(const optix::Ray &ray, const sphere &s,
                          float *thit=NULL, float3 *normal=NULL)
{
    float3 O = ray.origin - s.center;
    float3 D = ray.direction;

    float b = dot(O, D);
    float c = dot(O, O)-s.radius*s.radius;
    float disc = b*b-c;
    if(disc > 0.0f)
    {
	float sdisc = sqrtf(disc);
	float root1 = (-b - sdisc);

	if(root1 >= 0)
	{
            if(normal)
                *normal = (O + root1)*D/s.radius;
            if(thit)
                *thit = root1;
            return true;
	} 
        else
        {
	    float root2 = (-b + sdisc);
            if(root2 >= 0)
	    {
                if(normal)
                    *normal = (O + root2)*D/s.radius;
                if(thit)
                    *thit = root2;
                return true;
	    }
	}
    }
    return false;
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

__device__ float3 sample_surface(const sphere &s, 
                                const float3 &p, float u1, float u2,
                                float *pdf)
{
    float3 dv = s.center - p;

    float dist2 = dot(dv,dv);

    float3 wc = normalize(dv),
           wcX, wcY;

    coordinate_system(wc, &wcX, &wcY);

    float sinThetaMax2 = s.radius*s.radius / dist2,
          cosThetaMax = sqrt(max(0.f, 1.f - sinThetaMax2));

    float3 dir = uniform_sample_cone(u1, u2, cosThetaMax, wcX, wcY, wc);

    optix::Ray r(p, dir, 1, 1e-4, 1);
    float thit;
    if(!intersect(r, s, &thit))
    {
        thit = dot(dv, normalize(dir));
        if(pdf)
            *pdf = 1;
    }
    else
    {
        if(pdf)
            *pdf = uniform_cone_pdf(cosThetaMax);
    }

    return p + thit*dir;
}

RT_PROGRAM void closest_hit()
{
    float3 N
        = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

    float3 P = ray.origin + t_hit*ray.direction;

    float3 c = make_float3(0.1);

#if MODE == NO_SHADOWS
    int num_lights = point_lights.size();
    for(int i=0; i<num_lights; ++i)
    {
        float kd = clamp(dot(N, normalize(point_lights[i].pos-P)), 0.f, 1.f);
        c += make_float3(1)*kd;
    }
#elif MODE == HARD_SHADOWS
    int num_lights = point_lights.size();
    for(int i=0; i<num_lights; ++i)
    {
        float3 L = point_lights[i].pos - P;

        optix::Ray shadow_ray(P, normalize(L), 1, scene_epsilon, length(L));

        RayPayload_shadow payload;
        payload.attenuation = point_lights[i].color;

        rtTrace(root_object, shadow_ray, payload);

        if(payload.attenuation.x != 0)
        {
            float kd = clamp(dot(N, normalize(L)), 0.f, 1.f);
            c += make_float3(1)*kd;
        }
    }
#elif MODE == SOFT_SHADOWS

    sphere s;
    s.center = point_lights[0].pos;
    s.radius = 0.005;

    float3 C = make_float3(1,1,1);

    curandStateXORWOW_t state;
    curand_init(fabs(P.x)*1e6+fabs(P.y)*1e6+fabs(P.z)*1e6, 0, 0, &state);

    for(int sx=0; sx<SQRT_SHADOW_STRATA; ++sx)
    {
        for(int sy=0; sy<SQRT_SHADOW_STRATA; ++sy)
        {
            for(int i=0; i<SHADOW_SAMPLES_PER_STRATUM; ++i)
            {
                float u1 = (curand_uniform(&state)+sx)/SQRT_SHADOW_STRATA,
                      u2 = (curand_uniform(&state)+sy)/SQRT_SHADOW_STRATA;

                float3 ps = sample_surface(s, P, u1, u2, NULL);

                float3 L = ps-P;

                optix::Ray shadow_ray = optix::make_Ray(P, L, 1, scene_epsilon, 1);

                RayPayload_shadow payload;
                payload.attenuation = C;

                rtTrace(root_object, shadow_ray, payload);

                if(payload.attenuation.x != 0)
                {
                    float kd = clamp(dot(N, normalize(L)), 0.f, 1.f);
                    c += C*kd/SHADOW_TOTAL_SAMPLES;
                }
            }
        }
    }
#endif

    payload_radiance.result = c;
}

rtDeclareVariable(RayPayload_shadow, payload_shadow, rtPayload,);

RT_PROGRAM void any_hit_shadow()
{
    payload_shadow.attenuation = make_float3(0);
    rtTerminateRay();
}
