#ifndef MATERIAL_H
#define MATERIAL_H

namespace shade
{

class Material
{
public:
    __device__
    Material(float3 _color, float3 _specular_color, 
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
