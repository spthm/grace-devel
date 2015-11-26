#pragma once

#include <thrust/host_vector.h>

struct AABB
{
    float bx, by, bz;
    float tx, ty, tz;
};

void random_aabbs(thrust::host_vector<AABB>&, float box_min, float box_max);
