#pragma once

#include "grace/types.h"

namespace grace {

GRACE_ALIGNED_STRUCT(16) Ray
{
    float dx, dy, dz;
    float ox, oy, oz;
    float start;
    float end;
};

} //namespace grace
