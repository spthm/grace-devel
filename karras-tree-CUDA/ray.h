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

struct SlopeProp
{
    float xbyy, xbyz, ybyx, ybyz, zbyx, zbyy;
    float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
};

} //namespace grace
