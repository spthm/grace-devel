#pragma once

#include "vector3_common.h"

#include <stddef.h>

void test_vector3_size_host();

void test_vector3_padding_host();

void test_vector3_accessors_stack_host();

void test_vector3_accessors_heap_host();

Vector3Ptrs alloc_vector3s_host(const size_t n);

void free_vectors_host(const Vector3Ptrs);

void fill_vector3s_host(const Vector3Ptrs);
