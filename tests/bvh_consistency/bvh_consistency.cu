// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cpp/bvh.h"
#include "grace/cuda/bvh.cuh"
#include "grace/aabb.h"
#include "grace/sphere.h"
#include "helper/random.cuh"
#include "helper/tree.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

typedef grace::Sphere<float> SphereType;

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);

    int max_per_leaf = 32;
    size_t N = 128 * 128 * 128;
    unsigned int device_ID = 0;

    if (argc > 1) {
        max_per_leaf = (int)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N = (size_t)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        device_ID = (unsigned int)std::strtol(argv[3], NULL, 10);
    }

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    std::cout << "Max particles per leaf:   " << max_per_leaf << std::endl
              << "Number of particles:      " << N << std::endl
              << "Running on device:        " << device_ID
                                            << " (" << deviceProp.name << ")"
                                            << std::endl
              << std::endl;

    SphereType high = SphereType(1.0f, 1.0f, 1.0f, 0.1f);
    SphereType low = SphereType(-1.0f, -1.0f, -1.0f, 0.0f);

    thrust::device_vector<SphereType> d_spheres(N);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres.begin(),
                      random_sphere_functor<SphereType>(low, high));
    grace::CudaBvh d_tree(N, max_per_leaf);
    build_tree(d_spheres, d_tree);

    grace::HostBvh h_tree(N, max_per_leaf);
    d_tree.to_host(h_tree);
    const int N_leaves = h_tree.num_leaves();
    const int N_nodes = N_leaves - 1;

    thrust::host_vector<int> parent_flags(N_nodes);
    thrust::host_vector<int> child_flags(N_nodes + N_leaves);
    thrust::host_vector<int> particle_flags(N);

    typedef grace::detail::Bvh_ref<grace::HostBvh> ref_type;
    ref_type bvh_ref(h_tree);
    for (size_t ni = 0; ni < N_nodes; ++ni)
    {
        ref_type::node_type node = bvh_ref.nodes()[ni];
        int l = node.left_child();
        int r = node.right_child();

        parent_flags[ni] += 1;
        child_flags[l] += 1;
        child_flags[r] += 1;
    }

    for (size_t li = 0; li < N_leaves; ++li)
    {
        ref_type::leaf_type leaf = bvh_ref.leaves()[li];
        int first = leaf.first_primitive();
        int size = leaf.size();

        for (int pi = first; pi < first + size; ++pi)
        {
            particle_flags[pi] += 1;
        }
    }

    size_t failures = 0;

    for (size_t ni = 0; ni < parent_flags.size(); ++ni)
    {
        int flag = parent_flags[ni];
        if (flag != 1) {
            std::cout << "Error: parent count @ " << ni << " = " << flag
                      << std::endl;
            failures += 1;
        }
    }

    for (size_t ci = 0; ci < child_flags.size(); ++ci)
    {
        int flag = child_flags[ci];

        if (ci == h_tree.root_index() && flag != 0)
        {
            std::cout << "Error: child count @ root " << ci << " = " << flag
                      << std::endl;
            failures += 1;
        }

        if (ci != h_tree.root_index() && flag != 1) {
            std::cout << "Error: child count @ " << ci << " = " << flag
                      << std::endl;
            failures += 1;
        }
    }

    for (size_t pi = 0; pi < N; ++pi)
    {
        int flag = particle_flags[pi];
        if (flag != 1) {
            std::cout << "Error: particle count @ " << pi << " = " << flag
                      << std::endl;
            failures += 1;
        }
    }

    if (failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
}
