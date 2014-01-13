#pragma once

#include "types.h"

namespace grace {

struct Ray
{
    float dx, dy, dz;
    float ox, oy, oz;
    float length;
    int dclass;
};

} //namespace grace
