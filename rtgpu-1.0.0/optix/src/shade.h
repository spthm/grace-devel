#ifndef SHADE_H
#define SHADE_H

#include <optix.h>
#include "light.h"


rtDeclareVariable(float3, eye, ,);
rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);
rtDeclareVariable(float, t_hit, rtIntersectionDistance,);
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal,);
rtDeclareVariable(float3, shading_normal, attribute shading_normal,);
rtDeclareVariable(RayPayload_radiance, payload_radiance, rtPayload,);
rtBuffer<shade::point_light> point_lights;
rtDeclareVariable(rtObject, root_object,,);
rtDeclareVariable(float, scene_epsilon,,);

template <class M> __device__
inline void calc_shade(const M &mat)
{
    float3 world_geometric_normal 
        = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    float3 world_shading_normal 
        = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

    shade::context ctx;

    ctx.P = ray.origin + t_hit*ray.direction;
    ctx.N = faceforward(world_shading_normal, -ray.direction, 
                        world_geometric_normal);
    ctx.cosa = 0; // 180°

    ctx.V = eye - ctx.P;

    float3 result = make_float3(0);

    int num_lights = point_lights.size();
    for(int i=0; i<num_lights; ++i)
    {
        result += ctx.mat.color*ctx.mat.ka*point_lights[i].color;

        if(point_lights[i].casts_shadow)
        {
            float3 L = point_lights[i].pos - ctx.P;

            optix::Ray shadow_ray(ctx.P, normalize(L), 1, 
                                  scene_epsilon, length(L));

            RayPayload_shadow payload;
            payload.attenuation = point_lights[i].color;

            rtTrace(root_object, shadow_ray, payload);

            if(fmaxf(payload.attenuation) > 0)
                result += mat.luminance(point_lights[i], ctx)*payload.attenuation;
        }
        else
            result += mat.luminance(point_lights[i], ctx);
    }

    // now take care of reflections

    float power = payload_radiance.power * ctx.mat.ks;

    if(power > 0.01 && payload_radiance.depth < 3)
    {
        RayPayload_radiance payload_refl;
        payload_refl.power = power;
        payload_refl.depth = payload_radiance.depth+1;
        payload_refl.result = make_float3(0);

        float3 R = reflect(ray.direction, ctx.N);

        optix::Ray refl_ray(ctx.P, R, 0, scene_epsilon);

        rtTrace(root_object, refl_ray, payload_refl);

        result += ctx.mat.ks * payload_refl.result;
    }

#if 0
    power = payload_radiance.power * ctx.mat.kt;

    if(power > 0.01)
    {
        real ir = 
#endif


    payload_radiance.result = result;
}

rtDeclareVariable(RayPayload_shadow, payload_shadow, rtPayload,);

RT_PROGRAM void any_hit_shadow()
{
    payload_shadow.attenuation = make_float3(0);
    rtTerminateRay();
}

#endif
