#pragma once

#include "grace/ray.h"

#include <thrust/device_vector.h>

float resultant_length_squared(const thrust::device_vector<grace::Ray>& rays);

void An_Gn_statistics(const thrust::device_vector<grace::Ray>& rays,
                      double* An, double* Gn);
