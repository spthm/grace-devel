#ifndef PAYLOAD_H
#define PAYLOAD_H

struct RayPayload_radiance
{
    float3 result;
    float power;
    int depth;
};

struct RayPayload_shadow
{
    float3 attenuation;
};


#endif
