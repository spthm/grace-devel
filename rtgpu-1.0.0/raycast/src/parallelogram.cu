#include <optix.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>

rtDeclareVariable(float4, plane, , );
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
{
    float3 n = make_float3( plane );
    float dt = dot(ray.direction, n );
    float t = (plane.w - dot(n, ray.origin))/dt;
    if( t > ray.tmin && t < ray.tmax ) 
    {
	float3 p = ray.origin + ray.direction * t;
	float3 vi = p - anchor;
	float a1 = dot(v1, vi);
	if(a1 >= 0 && a1 <= 1)
	{
	    float a2 = dot(v2, vi);
	    if(a2 >= 0 && a2 <= 1)
	    {
		if( rtPotentialIntersection( t ) ) 
		{
		    geometric_normal = shading_normal = n;
		    texcoord = make_float3(a1,a2,0);
		    rtReportIntersection( 0 );
		}
	    }
	}
    }
}

RT_PROGRAM void bounds (int, float result[6])
{
    // v1 and v2 are scaled by 1./length^2. 
    // Rescale back to normal for the bounds computation.
    float3 tv1 = v1 / dot( v1, v1 );
    float3 tv2 = v2 / dot( v2, v2 );
    float3 p00 = anchor;
    float3 p01 = anchor + tv1;
    float3 p10 = anchor + tv2;
    float3 p11 = anchor + tv1 + tv2;

    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
    aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
}
