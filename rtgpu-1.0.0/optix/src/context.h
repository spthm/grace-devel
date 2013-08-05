#ifndef SHADE_CONTEXT_H
#define SHADE_CONTEXT_H

#include "material.h"
#include "cone.h"

namespace shade
{

struct context
{
    float3 P,N,V;
    material mat;
    float cosa;
};

}

#endif
