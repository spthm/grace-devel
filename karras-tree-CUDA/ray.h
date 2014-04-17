#pragma once

namespace grace {

struct Ray
{
    float4 dir; // dx, dy, dz, dclass (dclass is stored via __int_as_float!)
    float4 orig; // ox, oy, oz, length
};

struct RaySlope
{
    float xbyy, ybyx, ybyz, zbyy, xbyz, zbyx;
    float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
};

} //namespace grace
