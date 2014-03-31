#include <cstring>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "utils.cuh"
#include "../nodes.h"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);


    /* Initialize run parameters. */

    unsigned int device_ID = 0;
    unsigned int N_iter = 100;
    unsigned int start = 20;
    unsigned int end = 20;
    unsigned int seed_factor = 1u;

    if (argc > 1) {
        device_ID = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_iter = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        end = (unsigned int) std::strtol(argv[3], NULL, 10);
        end = min(28, max(5, end));
        if (end < start)
            start = end;
    }
    if (argc > 4) {
        end = (unsigned int) std::strtol(argv[4], NULL, 10);
        // Keep levels in [5, 28].
        end = min(28, max(5, end));
        start = (unsigned int) std::strtol(argv[3], NULL, 10);
        start = min(28, max(5, start));
    }
    if (argc > 5) {
        seed_factor = (unsigned int) std::strtol(argv[5], NULL, 10);
    }


    /* Output run parameters and device properties to console. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    std::cout << "Device " << device_ID
                    << ":                   " << deviceProp.name << std::endl;
    std::cout << "MORTON_THREADS_PER_BLOCK:   " << MORTON_THREADS_PER_BLOCK
            << std::endl;
    std::cout << "BUILD_THREADS_PER_BLOCK:    " << BUILD_THREADS_PER_BLOCK
            << std::endl;
    std::cout << "AABB_THREADS_PER_BLOCK:     " << AABB_THREADS_PER_BLOCK
            << std::endl;
    std::cout << "MAX_BLOCKS:                 " << MAX_BLOCKS << std::endl;
    std::cout << "Starting tree depth:        " << start << std::endl;
    std::cout << "Finishing tree depth:       " << end << std::endl;
    std::cout << "Iterations per tree:        " << N_iter << std::endl;
    std::cout << "Random points' seed factor: " << seed_factor << std::endl;
    std::cout << std::endl << std::endl;


    /* Profile the tree by generating random data, and further profile the AABB
     * construction with an additional fully-balanced tree built on the host.
     */

    for (int levels=start; levels<=end; levels++)
    {
        unsigned int N = 1u << (levels - 1);


        /* Generate N random points as floats in [0,1) and radii in [0,0.1). */

        thrust::host_vector<float4> h_spheres_xyzr(N);
        thrust::transform(thrust::counting_iterator<unsigned int>(0),
                          thrust::counting_iterator<unsigned int>(N),
                          h_spheres_xyzr.begin(),
                          grace::random_float4_functor(0.1f, seed_factor));

        // Set the tree-build AABB (contains all sphere centres).
        float4 bottom = make_float4(0., 0., 0., 0.);
        float4 top = make_float4(1., 1., 1., 0.);


        /* Build the tree from and time it for N_iter iterations. */

        cudaEvent_t part_start, part_stop;
        cudaEvent_t tot_start, tot_stop;
        float part_elapsed;
        double all_tot, morton_tot, sort_tot, tree_tot, aabb_tot;
        cudaEventCreate(&part_start);
        cudaEventCreate(&part_stop);
        cudaEventCreate(&tot_start);
        cudaEventCreate(&tot_stop);

        for (int i=0; i<N_iter; i++) {
            cudaEventRecord(tot_start);
            // Copy pristine host-side data to GPU.
            thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;

            thrust::device_vector<grace::uinteger32> d_keys(N);
            cudaEventRecord(part_start);
            grace::morton_keys(d_spheres_xyzr, d_keys, bottom, top);
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

            grace::Nodes d_nodes(N-1);
            grace::Leaves d_leaves(N);
            cudaEventRecord(part_start);
            grace::build_nodes(d_nodes, d_leaves, d_keys);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            tree_tot += part_elapsed;

            cudaEventRecord(part_start);
            grace::find_AABBs(d_nodes, d_leaves,
                              d_spheres_xyzr);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            aabb_tot += part_elapsed;

            // Record the total time spent in the loop.
            cudaEventRecord(tot_stop);
            cudaEventSynchronize(tot_stop);
            cudaEventElapsedTime(&part_elapsed, tot_start, tot_stop);
            all_tot += part_elapsed;
        }

        std::cout << "Will generate:" << std::endl;
        std::cout << "    i)  A tree from " << N << " random points." << std::endl;
        std::cout << "    ii) A fully-balanced tree with " << levels
                << " levels and " << N << " leaves." << std::endl;
        std::cout << std::endl;
        std::cout << "Time for Morton key generation:   ";
        std::cout.width(7);
        std::cout << morton_tot/N_iter << " ms." << std::endl;
        std::cout << "Time for sort-by-key:             ";
        std::cout.width(7);
        std::cout << sort_tot/N_iter << " ms." << std::endl;
        std::cout << "Time for hierarchy generation:    ";
        std::cout.width(7);
        std::cout << tree_tot/N_iter << " ms." << std::endl;
        std::cout << "Time for calculating AABBs:       ";
        std::cout.width(7);
        std::cout << aabb_tot/N_iter << " ms." << std::endl;
        std::cout << "Time for total (inc. memory ops): ";
        std::cout.width(7);
        std::cout << all_tot/N_iter << " ms." << std::endl;


        /* Build fully-balanced tree on host. */

        grace::H_Nodes h_nodes(N-1);
        grace::H_Leaves h_leaves(N);

        // Set up bottom level (where all nodes connect to leaves).
        for (unsigned int i_left=1; i_left<N-1; i_left+=4)
        {
            unsigned int i_right = i_left + 1;

            h_nodes.hierarchy[i_left].x = i_left - 1 + N-1;
            h_nodes.hierarchy[i_left].y = i_left + N-1;
            h_nodes.hierarchy[i_left].w = i_left - 1;

            h_nodes.hierarchy[i_right].x = i_right + N-1;
            h_nodes.hierarchy[i_right].y = i_right + 1 +N-1;
            h_nodes.hierarchy[i_right].w = i_right + 1;

            h_leaves.parent[i_left-1] = h_leaves.parent[i_left] = i_left;
            h_leaves.parent[i_right] = h_leaves.parent[i_right+1] = i_right;
        }

        // Set up all except bottom and top levels, starting at bottom-but-one.
        for (unsigned int height=2; height<(levels-1); height++)
        {
            for (unsigned int i_left=(1u<<height)-1;
                              i_left<N-1;
                              i_left+=1u<<(height+1))
            {
                unsigned int i_right = i_left + 1;
                unsigned int i_left_split = (2*i_left - (1u<<height)) / 2;
                unsigned int i_right_split = i_left_split + (1u<<height);

                h_nodes.hierarchy[i_left].x = i_left_split;
                h_nodes.hierarchy[i_left].y = i_left_split + 1;
                h_nodes.hierarchy[i_left].w = i_left - (1u<<height) + 1;

                h_nodes.hierarchy[i_right].x = i_right_split;
                h_nodes.hierarchy[i_right].y = i_right_split + 1;
                h_nodes.hierarchy[i_right].w = i_right + (1u<<height) - 1;

                h_nodes.hierarchy[i_left_split].z =
                    h_nodes.hierarchy[i_left_split+1].z = i_left;
                h_nodes.hierarchy[i_right_split].z =
                    h_nodes.hierarchy[i_right_split+1].z = i_right;
            }
        }

        // Set up root node and link children to it.
        h_nodes.hierarchy[0].x = N/2 - 1;
        h_nodes.hierarchy[0].y = N/2;
        h_nodes.hierarchy[N/2 - 1].z = h_nodes.hierarchy[N/2].z = 0;
        h_nodes.hierarchy[0].w = N - 1;


        /* Profile the fully-balanced tree. */

        grace::Nodes d_nodes(N-1);
        grace::Leaves d_leaves(N);
        part_elapsed = 0;
        aabb_tot = 0;
        for (int i=0; i<N_iter; i++)
        {
            // NB: Levels and AABBs do not need copying: we don't build them on
            // the host.
            d_nodes.hierarchy = h_nodes.hierarchy;
            thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;

            cudaEventRecord(part_start);
            grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            aabb_tot += part_elapsed;
        }

        aabb_tot /= N_iter;
        std::cout << "Time for balanced tree AABBs:     ";
        std::cout.width(7);
        std::cout << aabb_tot << " ms." << std::endl;
        std::cout << std::endl << std::endl;
    }
}
