#include "vector4_host.h"
#include "vector4_device.h"

int main(void)
{
    // Host tests

    test_vector4_size_host();

    test_vector4_padding_host();

    test_vector4_accessors_stack_host();

    test_vector4_accessors_heap_host();


    // Device tests

    // Triggering device-side assert may not alter the exit status.
    int status1, status2;

    status1 = test_vector4_accessors_stack_device();

    Vector4Ptrs vec4_ptrs_host = alloc_vector4s_host(NUM_TEST_VECTORS);
    Vector4Ptrs vec4_ptrs_device = alloc_vector4s_device(NUM_TEST_VECTORS);

    fill_vector4s_host(vec4_ptrs_host);
    fill_vector4s_device(vec4_ptrs_device);

    status2 = compare_vector4s_host_to_device(vec4_ptrs_host, vec4_ptrs_device);

    free_vectors_host(vec4_ptrs_host);
    free_vectors_device(vec4_ptrs_device);

    return status1 | status2;
}
