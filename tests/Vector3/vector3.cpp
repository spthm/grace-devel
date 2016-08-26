#include "vector3_host.h"
#include "vector3_device.h"

int main(void)
{
    // Host tests

    test_vector3_size_host();

    test_vector3_padding_host();

    test_vector3_accessors_stack_host();

    test_vector3_accessors_heap_host();

    // Device tests

    // Triggering device-side assert may not alter the exit status.
    int status1, status2;

    status1 = test_vector3_accessors_stack_device();

    Vector3Ptrs vec3_ptrs_host = alloc_vector3s_host(NUM_TEST_VECTORS);
    Vector3Ptrs vec3_ptrs_device = alloc_vector3s_device(NUM_TEST_VECTORS);

    fill_vector3s_host(vec3_ptrs_host);
    fill_vector3s_device(vec3_ptrs_device);

    status2 = compare_vector3s_host_to_device(vec3_ptrs_host, vec3_ptrs_device);

    free_vectors_host(vec3_ptrs_host);
    free_vectors_device(vec3_ptrs_device);

    return status1 | status2;
}
