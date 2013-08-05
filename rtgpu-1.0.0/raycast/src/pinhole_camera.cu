#include <optix.h>
#include <optix_math.h>

#include "payload.h"

rtDeclareVariable(float3, eye,,);
rtDeclareVariable(float3, U,,);
rtDeclareVariable(float3, V,,);
rtDeclareVariable(float3, W,,);
rtDeclareVariable(uint2, launch_index, rtLaunchIndex,);
rtDeclareVariable(float, scene_epsilon,,);

rtDeclareVariable(rtObject, root_object,,);

rtBuffer<float4, 2> output_buffer;

RT_PROGRAM void pinhole_camera()
{
    uint2 screen = output_buffer.size();

    float2 d = make_float2(launch_index) / make_float2(screen)*2 - 1;
    
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);

    optix::Ray ray(ray_origin, ray_direction, 0, scene_epsilon);

    RayPayload_radiance payload;
    payload.depth = 0;
    payload.power = 1;
    payload.result = make_float3(0);

    rtTrace(root_object, ray, payload);

    output_buffer[launch_index] = make_float4(payload.result,1);
}

RT_PROGRAM void miss()
{
    output_buffer[launch_index] = make_float4(0,0,1,1);
}

RT_PROGRAM void exception()
{
    const unsigned int code = rtGetExceptionCode();
    rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", 
	     code, launch_index.x, launch_index.y);

    output_buffer[launch_index] = make_float4(1,0,0,1);
}
