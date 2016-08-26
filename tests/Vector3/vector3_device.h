#pragma once

#include "vector3_common.h"

#include <stddef.h>

int test_vector3_accessors_stack_device();

int compare_vector3s_host_to_device(const Vector3Ptrs ref_vec3_ptrs_host,
                                    const Vector3Ptrs vec3_ptrs_device);

Vector3Ptrs alloc_vector3s_device(const size_t n);

void free_vectors_device(const Vector3Ptrs vec_ptrs_device);

void fill_vector3s_device(const Vector3Ptrs vec_ptrs_device);
