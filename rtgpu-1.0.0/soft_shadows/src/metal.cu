template <class L>
__device__
float3 metal_luminance(const L &light, const material &mat)
{
    if(ka > 0)
    {
	amb = ambient(light, ctx);
	amb.color *= ctx.mat.ka;
	amb.intensity *= ctx.mat.ka;
    }
    else
    {
	amb.color = make_float3(0,0,0);
	amb.intensity = 0;
    }

    if(ks > 0)
    {
	spec = specular(light, ctx);
	spec.color *= ctx.mat.ks;
	spec.intensity *= ctx.mat.ks;
    }
    else
    {
	spec.color = make_float3(0,0,0);
	spec.intensity = 0;
    }

    return illumination(color*(amb.color+spec.color)
			amb.intensity + spec.intensity);
}

