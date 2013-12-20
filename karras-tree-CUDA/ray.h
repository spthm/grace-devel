#pragma once

#include "types.h"

namespace grace {

struct Ray
{
    float dx, dy, dz;
    float ox, oy, oz;
    float xbyy, ybyx, ybyz, zbyy, zbyx, xbyz;
    float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
    float length;
    int dclass;
};

} //namespace grace
