#ifndef SHADE_LIGHT_H
#define SHADE_LIGHT_H

#include <optix_math.h>
#include "context.h"

namespace shade
{

struct light_ray
{
    float3 dir;
    float3 color;
};

struct light
{
    __device__
    light(float3 _color, float _intensity, float _ka)
	: color(_color), ka(_ka), intensity(_intensity) {}

    float3 color;
    float intensity, ka;

    bool casts_shadow;
};

struct point_light : light
{
    __device__
    point_light(float3 color, float intensity, float3 _pos, float ka=.1,
		float att0=0, float att1=0, float att2=0)
	: light(color, intensity, ka)
	, pos(_pos)
    {
	att[0] = att0;
	att[1] = att1;
	att[2] = att2;
    }

    __device__
    float attenuation(float dist) const
    {
	return att[0] + att[1]*dist + att[2]*dist*dist;
    }

    __device__
    bool transport(light_ray &ray, const context &ctx) const
    {
	float3 v = ctx.P - pos;
	float dist = length(v);

	float3 dir = v/dist; // unit(v)

        cone recv;
        recv.origin = ctx.P;
        recv.dir = ctx.N;
        recv.cosa = ctx.cosa;

	if(recv.is_coupled_to(dir))
	{
	    float d = attenuation(dist);
	    float att = d > 0 ? 1/d : 1;

            ray.dir = -dir;
            ray.color = att*intensity*color;
            return true;
	}
	else
            return false;
    }

    float3 pos;
    float att[3];
};

} // namespace shade

#endif
