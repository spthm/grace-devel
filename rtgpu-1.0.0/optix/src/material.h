#ifndef MATERIAL_H
#define MATERIAL_H

#include <optix.h>

#ifdef __CUDACC__
rtDeclareVariable(float3, color, ,);
rtDeclareVariable(float3, specular_color, ,);
rtDeclareVariable(float, kd, ,);
rtDeclareVariable(float, ka, ,);
rtDeclareVariable(float, ks, ,);
rtDeclareVariable(float, kt, ,);
rtDeclareVariable(float, shininess, ,);
rtDeclareVariable(float, ior, ,);
#endif

namespace shade
{

class material
{
public:
#ifdef __CUDACC__
    __device__ material()
	: color(::color)
	, specular_color(::specular_color)
	, ka(::ka)
	, kd(::kd)
	, ks(::ks)
	, shininess(::shininess)
	, kt(::kt)
	, ior(::ior)
    {
    }
#endif

    __device__
    material(float3 _color, float3 _specular_color, 
	     float _ka, float _kd, float _ks,
	     float _shininess, float _kt, float _ior)
	: color(_color)
	, specular_color(_specular_color)
	, ka(_ka)
	, kd(_kd)
	, ks(_ks)
	, shininess(_shininess)
	, kt(_kt)
	, ior(_ior)
    {
    }

    float3 color;
    float3 specular_color;
    float kd, ka, ks;

    float kt, ior, shininess;
};

}

#endif
