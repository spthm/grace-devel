#ifndef SHADE_CONTEXT_H
#define SHADE_CONTEXT_H

#include "material.h"
#include "cone.h"

namespace shade
{

struct Context
{
    float3 P,N,V;
    Material mat;
    float cosa;
};

}

#endif
