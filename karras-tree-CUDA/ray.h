#pragma once

namespace grace {

struct Ray
{
    float dx, dy, dz;
    float ox, oy, oz;
    float length;
    int dclass;
};

struct RaySlope
{
    float xbyy, ybyx, ybyz, zbyy, xbyz, zbyx;
    float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
};

} //namespace grace
