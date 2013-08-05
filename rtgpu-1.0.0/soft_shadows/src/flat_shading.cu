#include <optix.h>
#include "payload.h"

rtDeclareVariable(float3, color,,);
rtDeclareVariable(RayPayload_radiance, payload, rtPayload,);

RT_PROGRAM void closest_hit()
{
    payload.result = color;
}
