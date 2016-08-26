#pragma once

#include "vector4_common.h"

#include <stddef.h>

void test_vector4_size_host();

void test_vector4_padding_host();

void test_vector4_accessors_stack_host();

void test_vector4_accessors_heap_host();

Vector4Ptrs alloc_vector4s_host(const size_t n);

void free_vectors_host(const Vector4Ptrs);

void fill_vector4s_host(const Vector4Ptrs);
