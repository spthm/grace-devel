#pragma once

#include "vector4_common.h"

#include <stddef.h>

int test_vector4_accessors_stack_device();

int compare_vector4s_host_to_device(const Vector4Ptrs ref_vec4_ptrs_host,
                                    const Vector4Ptrs vec4_ptrs_device);

Vector4Ptrs alloc_vector4s_device(const size_t n);

void free_vectors_device(const Vector4Ptrs vec_ptrs);

void fill_vector4s_device(const Vector4Ptrs vec_ptrs_device);
