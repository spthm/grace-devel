#pragma once

#include "grace/config.h"

namespace grace {

struct GRACE_ALIGNAS(16) Ray
{
    float dx, dy, dz;
    float ox, oy, oz;
    float start;
    float end;
};

} //namespace grace
