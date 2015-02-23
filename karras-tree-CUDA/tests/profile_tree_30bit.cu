#include <cstring>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../utils.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/morton.cuh"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);


    /* Initialize run parameters. */

    unsigned int device_ID = 0;
    unsigned int max_per_leaf = 32;
    unsigned int N_iter = 100;
    unsigned int start = 20;
    unsigned int end = 23;
    unsigned int seed_factor = 1u;

    if (argc > 1) {
        device_ID = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        max_per_leaf = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
    }
    if (argc > 4) {
        end = (unsigned int) std::strtol(argv[4], NULL, 10);
        end = min(28, max(5, end));
        if (end < start)
            start = end;
    }
    if (argc > 5) {
        end = (unsigned int) std::strtol(argv[5], NULL, 10);
        // Keep levels in [5, 28].
        end = min(28, max(5, end));
        start = (unsigned int) std::strtol(argv[4], NULL, 10);
        start = min(28, max(5, start));
    }
    if (argc > 6) {
        seed_factor = (unsigned int) std::strtol(argv[6], NULL, 10);
    }


    /* Output run parameters and device properties to console. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    std::cout << "Device " << device_ID
                    << ":                   " << deviceProp.name << std::endl;
    std::cout << "MORTON_THREADS_PER_BLOCK:   "
              << grace::MORTON_THREADS_PER_BLOCK << std::endl;
    std::cout << "BUILD_THREADS_PER_BLOCK:    "
              << grace::BUILD_THREADS_PER_BLOCK << std::endl;
    std::cout << "AABB_THREADS_PER_BLOCK:     "
              << grace::AABB_THREADS_PER_BLOCK << std::endl;
    std::cout << "MAX_BLOCKS:                 "
              << grace::MAX_BLOCKS << std::endl;
    std::cout << "Starting log2(N_points):    " << start << std::endl;
    std::cout << "Finishing log2(N_points):   " << end << std::endl;
    std::cout << "Max points per leaf:        " << max_per_leaf << std::endl;
    std::cout << "Iterations per tree:        " << N_iter << std::endl;
    std::cout << "Random points' seed factor: " << seed_factor << std::endl;
    std::cout << std::endl << std::endl;


    /* Profile the tree by generating random data, and further profile the AABB
     * construction with an additional fully-balanced tree built on the host.
     */

    for (int power = start; power <= end; power++)
    {
        unsigned int N = 1u << power;


        /* Generate N random points as floats in [0,1) and radii in [0,0.1). */

        thrust::host_vector<float4> h_spheres_xyzr(N);
        thrust::transform(thrust::counting_iterator<unsigned int>(0),
                          thrust::counting_iterator<unsigned int>(N),
                          h_spheres_xyzr.begin(),
                          grace::random_float4_functor(0.1f, seed_factor));

        // Set the tree-build AABB (contains all sphere centres).
        float3 top = make_float3(1., 1., 1.);
        float3 bot = make_float3(0., 0., 0.);


        /* Build the tree from and time it for N_iter iterations. */

        cudaEvent_t part_start, part_stop;
        cudaEvent_t tot_start, tot_stop;
        float part_elapsed;
        double all_tot, morton_tot, sort_tot;
        double deltas_tot, leaves_tot, leaf_deltas_tot, nodes_tot;
        cudaEventCreate(&part_start);
        cudaEventCreate(&part_stop);
        cudaEventCreate(&tot_start);
        cudaEventCreate(&tot_stop);

        for (int i=0; i<N_iter; i++) {
            cudaEventRecord(tot_start);

            thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;
            thrust::device_vector<grace::uinteger32> d_keys(N);

            cudaEventRecord(part_start);
            grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            morton_tot += part_elapsed;

            cudaEventRecord(part_start);
            thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                                d_spheres_xyzr.begin());
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            sort_tot += part_elapsed;

            thrust::device_vector<float> d_deltas(N+1);

            cudaEventRecord(part_start);
            grace::compute_deltas(d_spheres_xyzr, d_deltas);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            deltas_tot += part_elapsed;

            grace::Tree d_tree(N, max_per_leaf);
            thrust::device_vector<int2> d_tmp_nodes(N-1);

            cudaEventRecord(part_start);
            grace::build_leaves(d_tmp_nodes, d_tree.leaves, d_tree.max_per_leaf,
                                d_deltas);
            grace::remove_empty_leaves(d_tree);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            leaves_tot += part_elapsed;

            const size_t n_new_leaves = d_tree.leaves.size();
            thrust::device_vector<float> d_new_deltas(n_new_leaves + 1);

            cudaEventRecord(part_start);
            grace::compute_leaf_deltas(d_tree.leaves, d_spheres_xyzr,
                                       d_new_deltas);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            leaf_deltas_tot += part_elapsed;

            cudaEventRecord(part_start);
            grace::build_nodes(d_tree, d_new_deltas, d_spheres_xyzr);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            nodes_tot += part_elapsed;

            // Record the total time spent in the loop.
            cudaEventRecord(tot_stop);
            cudaEventSynchronize(tot_stop);
            cudaEventElapsedTime(&part_elapsed, tot_start, tot_stop);
            all_tot += part_elapsed;
        }

        std::cout << "Will generate a tree from " << N << " random points."
                  << std::endl;
        std::cout << std::endl;

        std::cout << "Time for Morton key generation:    ";
        std::cout.width(7);
        std::cout << morton_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for sort-by-key:              ";
        std::cout.width(7);
        std::cout << sort_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for computing deltas:         ";
        std::cout.width(7);
        std::cout << deltas_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for building leaves:          ";
        std::cout.width(7);
        std::cout << leaves_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for computing leaf deltas:    ";
        std::cout.width(7);
        std::cout << leaf_deltas_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for building nodes:           ";
        std::cout.width(7);
        std::cout << nodes_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for total (inc. memory ops):  ";
        std::cout.width(7);
        std::cout << all_tot/N_iter << " ms." << std::endl;
        std::cout << std::endl << std::endl;
    }

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
