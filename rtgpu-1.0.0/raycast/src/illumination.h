#ifndef SHADE_ILLUMINATION_H
#define SHADE_ILLUMINATION_H

#include "light.h"

namespace shade
{

template <class L> __device__ 
inline float3 ambient(const L &light, const context &)
{
    return light.ka * light.color;
}

template <class L> __device__ 
inline float3 diffuse(const L &light, const context &ctx)
{
    light_ray ray;
    if(light.transport(ray, ctx))
    {
	float factor = dot(ray.dir, ctx.N);
	return factor*ray.color;
    }
    return make_float3(0,0,0);
}

template <class L> __device__
inline float3 specular(const L &light, const context &ctx)
{
    light_ray ray;
    if(light.transport(ray, ctx));
    {
	float3 H = normalize(light.pos - ctx.P + ctx.V);

	float fspec = dot(H, ctx.N);
	if(fspec > 0)
	{
	    float coef = pow(fspec, ctx.mat.shininess);
	    return coef*ray.color;
	}
    }
    return make_float3(0,0,0);
}

}

#endif
