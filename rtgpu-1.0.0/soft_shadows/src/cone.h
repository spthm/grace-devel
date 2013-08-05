#ifndef SHADE_CONE_H
#define SHADE_CONE_H

#include <optix_math.h>

namespace shade
{

struct Cone
{
    float3 origin;
    float3 dir;
    float cosa;

    __device__
    bool is_coupled_to(const float3 &v) const
    {
	return dot(dir, -v) > cosa;
    }
};

__device__
inline bool are_coupled(const Cone &c1, const Cone &c2)
{
    float3 d = normalize(c1.origin - c2.origin);
    return c1.is_coupled_to(d) && c2.is_coupled_to(-d);
}

}

#endif
