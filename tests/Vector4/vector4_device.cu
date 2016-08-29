#include "vector4_device.h"

// Forward declarations. These are not externally visible.

static void test_vector4_accessors_stack_device_impl();
static void compare_vector4s_host_to_device_impl(const Vector4Ptrs, const Vector4Ptrs);
static void copy_vector4s_host_to_device(const Vector4Ptrs, const Vector4Ptrs);
static __global__ void fill_vector4s_kernel(const Vector4Ptrs);


// The externally visible funtions.

int test_vector4_accessors_stack_device()
{
    test_vector4_accessors_stack_device_impl();

    cudaError_t cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
        printf("FAILED device stack-allocated Vector<4, T> accessors\n  CUDA Error: %s\n",
               cudaGetErrorString(cuerr));
        return 1;
    }
    else {
        printf("PASSED device stack-allocated Vector<4, T> accessors\n");
        return 0;
    }
}

int compare_vector4s_host_to_device(const Vector4Ptrs vec4_ptrs_host,
                                    const Vector4Ptrs vec4_ptrs_device)
{
    compare_vector4s_host_to_device_impl(vec4_ptrs_host, vec4_ptrs_device);

    cudaError_t cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
        printf("FAILED device-host Vector<4, T> comparisons\n  CUDA Error: %s\n",
               cudaGetErrorString(cuerr));
        return 1;
    }
    else {
        printf("PASSED device-host Vector<4, T> comparisons\n");
        return 0;
    }
}

Vector4Ptrs alloc_vector4s_device(const size_t n)
{
    Vector4Ptrs ptrs;

    cudaMalloc((void**)&ptrs.vec_1b,  n * sizeof(vec4_1T));
    cudaMalloc((void**)&ptrs.vec_2b,  n * sizeof(vec4_2T));
    cudaMalloc((void**)&ptrs.vec_3b,  n * sizeof(vec4_3T));
    cudaMalloc((void**)&ptrs.vec_4b,  n * sizeof(vec4_4T));
    cudaMalloc((void**)&ptrs.vec_5b,  n * sizeof(vec4_5T));
    cudaMalloc((void**)&ptrs.vec_6b,  n * sizeof(vec4_6T));
    cudaMalloc((void**)&ptrs.vec_7b,  n * sizeof(vec4_7T));
    cudaMalloc((void**)&ptrs.vec_8b,  n * sizeof(vec4_8T));
    cudaMalloc((void**)&ptrs.vec_16b, n * sizeof(vec4_16T));

    ptrs.n = n;

    return ptrs;
}

void free_vectors_device(const Vector4Ptrs vec_ptrs)
{
    cudaFree(vec_ptrs.vec_1b);
    cudaFree(vec_ptrs.vec_2b);
    cudaFree(vec_ptrs.vec_3b);
    cudaFree(vec_ptrs.vec_4b);
    cudaFree(vec_ptrs.vec_5b);
    cudaFree(vec_ptrs.vec_6b);
    cudaFree(vec_ptrs.vec_7b);
    cudaFree(vec_ptrs.vec_8b);
    cudaFree(vec_ptrs.vec_16b);
}

void fill_vector4s_device(const Vector4Ptrs vec_ptrs_device) {
    fill_vector4s_kernel<<<1, 1>>>(vec_ptrs_device);
}

// Private functions.

__device__ void copy_vector4s(const Vector4Ptrs src,
                              const Vector4Ptrs dst,
                              const size_t index)
{
    *(dst.vec_1b + index)  = *(src.vec_1b  + index);
    *(dst.vec_2b + index)  = *(src.vec_2b  + index);
    *(dst.vec_3b + index)  = *(src.vec_3b  + index);
    *(dst.vec_4b + index)  = *(src.vec_4b  + index);
    *(dst.vec_5b + index)  = *(src.vec_5b  + index);
    *(dst.vec_6b + index)  = *(src.vec_6b  + index);
    *(dst.vec_7b + index)  = *(src.vec_7b  + index);
    *(dst.vec_8b + index)  = *(src.vec_8b  + index);
    *(dst.vec_16b + index) = *(src.vec_16b + index);
}

static __global__ void fill_vector4s_kernel(const Vector4Ptrs vec_ptrs_dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < vec_ptrs_dst.n; ++i)
        {
            set_vector4_data(vec_ptrs_dst, i, vec_ptrs_dst.n);
        }
    }
}

static __global__ void test_vector4_accessors_stack_kernel(
    const Vector4Ptrs vec_ptrs_dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Identical to test_vector4_accessors_stack_host() in vector_host.cpp.
        Vector4Ptrs vec_ptrs;
        vec_ptrs.n = NUM_TEST_VECTORS;
        vec4_1T  vec_1b_array[ NUM_TEST_VECTORS];
        vec4_2T  vec_2b_array[ NUM_TEST_VECTORS];
        vec4_3T  vec_3b_array[ NUM_TEST_VECTORS];
        vec4_4T  vec_4b_array[ NUM_TEST_VECTORS];
        vec4_5T  vec_5b_array[ NUM_TEST_VECTORS];
        vec4_6T  vec_6b_array[ NUM_TEST_VECTORS];
        vec4_7T  vec_7b_array[ NUM_TEST_VECTORS];
        vec4_8T  vec_8b_array[ NUM_TEST_VECTORS];
        vec4_16T vec_16b_array[NUM_TEST_VECTORS];

        vec_ptrs.vec_1b  = vec_1b_array;
        vec_ptrs.vec_2b  = vec_2b_array;
        vec_ptrs.vec_3b  = vec_3b_array;
        vec_ptrs.vec_4b  = vec_4b_array;
        vec_ptrs.vec_5b  = vec_5b_array;
        vec_ptrs.vec_6b  = vec_6b_array;
        vec_ptrs.vec_7b  = vec_7b_array;
        vec_ptrs.vec_8b  = vec_8b_array;
        vec_ptrs.vec_16b = vec_16b_array;

        for (int i = 0; i < vec_ptrs.n; ++i)
        {
            set_vector4_data(vec_ptrs, i, vec_ptrs.n);
        }

        for (int i = 0; i < vec_ptrs.n; ++i)
        {
            test_vector4_accessors(vec_ptrs, i);
        }

        // We need to write something to global memory, or nvcc will treat the
        // above as dead code.
        for (int i = 0; i < vec_ptrs.n; ++i)
        {
            copy_vector4s(vec_ptrs, vec_ptrs_dst, i);
        }
    }
}

static __global__ void compare_vector4s_kernel(const Vector4Ptrs vecA_ptrs,
                                               const Vector4Ptrs vecB_ptrs)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < vecA_ptrs.n; ++i)
        {
            compare_vector4s(vecA_ptrs, vecB_ptrs, i);
        }
    }
}

static void copy_vector4s_host_to_device(const Vector4Ptrs src_host,
                                         const Vector4Ptrs dst_device)
{
    cudaMemcpy(dst_device.vec_1b,  src_host.vec_1b,
               dst_device.n * sizeof(vec4_1T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_2b,  src_host.vec_2b,
               dst_device.n * sizeof(vec4_2T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_3b,  src_host.vec_3b,
               dst_device.n * sizeof(vec4_3T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_4b,  src_host.vec_4b,
               dst_device.n * sizeof(vec4_4T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_5b,  src_host.vec_5b,
               dst_device.n * sizeof(vec4_5T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_6b,  src_host.vec_6b,
               dst_device.n * sizeof(vec4_6T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_7b,  src_host.vec_7b,
               dst_device.n * sizeof(vec4_7T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_8b,  src_host.vec_8b,
               dst_device.n * sizeof(vec4_8T),  cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device.vec_16b, src_host.vec_16b,
               dst_device.n * sizeof(vec4_16T), cudaMemcpyHostToDevice);
}

static void test_vector4_accessors_stack_device_impl()
{
    Vector4Ptrs vec_ptrs = alloc_vector4s_device(NUM_TEST_VECTORS);
    test_vector4_accessors_stack_kernel<<<1, 1>>>(vec_ptrs);
    free_vectors_device(vec_ptrs);
}

static void compare_vector4s_host_to_device_impl(
    const Vector4Ptrs ref_vec_ptrs_host,
    const Vector4Ptrs vec_ptrs_device)
{
    Vector4Ptrs ref_vec_ptrs_device = alloc_vector4s_device(NUM_TEST_VECTORS);
    copy_vector4s_host_to_device(ref_vec_ptrs_host, ref_vec_ptrs_device);

    compare_vector4s_kernel<<<1, 1>>>(ref_vec_ptrs_device, vec_ptrs_device);

    free_vectors_device(ref_vec_ptrs_device);
}