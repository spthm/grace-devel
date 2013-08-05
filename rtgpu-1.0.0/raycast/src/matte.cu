
template <class L>
__device__
float3 matte_luminance(const L &light, const context &ctx)
{
    float3 amb, dif;

    if(ka > 0)
	amb = ambient(light, ctx) * ctx.mat.ka;
    else
	amb = make_float3(0,0,0);

    if(kd > 0)
	dif = diffuse(light, ctx) * ctx.mat.kd;
    else
	dif = make_float3(0,0,0);

    return ctx.mat.color*(amb+dif);
}

