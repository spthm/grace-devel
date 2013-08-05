#include <optix.h>

#include "context.h"
#include "payload.h"
#include "illumination.h"
#include "material.h"
#include "shade.h"

namespace shade
{

struct plastic
{
public:
    template <class L> __device__
    float3 luminance(const L &light, const context &ctx) const;
};

template <class L> __device__
inline float3 plastic::luminance(const L &light, const context &ctx) const
{
    float3  dif, spec;

#if 0
    if(ctx.mat.ka > 0)
	amb = ambient(light, ctx) * ctx.mat.ka;
    else
	amb = make_float3(0,0,0);
#endif

    if(ctx.mat.kd > 0)
	dif = diffuse(light, ctx) * ctx.mat.kd;
    else
	dif = make_float3(0,0,0);

    if(ctx.mat.ks > 0)
	spec = specular(light, ctx) * ctx.mat.ks;
    else
	spec = make_float3(0,0,0);

    return ctx.mat.color*(dif) + ctx.mat.specular_color*spec;
}

}

RT_PROGRAM void closest_hit()
{
    calc_shade(shade::plastic());
}
