#pragma once

#include "grace/ray.h"
#include "grace/vector.h"

#include <thrust/host_vector.h>

typedef grace::Ray Ray;

enum DIR_CLASS { MMM, PMM, MPM, PPM, MMP, PMP, MPP, PPP };
enum { MISS = 0, HIT = 1 };

void isotropic_rays(thrust::host_vector<Ray>&, grace::Vector<3, float>,
                    float length);
