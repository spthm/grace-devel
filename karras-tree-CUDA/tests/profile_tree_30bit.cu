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
    unsigned int max_per_leaf = 100;
    unsigned int N_iter = 100;
    unsigned int start = 20;
    unsigned int end = 20;
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
    std::cout << "MORTON_THREADS_PER_BLOCK:   " << MORTON_THREADS_PER_BLOCK
            << std::endl;
    std::cout << "BUILD_THREADS_PER_BLOCK:    " << BUILD_THREADS_PER_BLOCK
            << std::endl;
    std::cout << "AABB_THREADS_PER_BLOCK:     " << AABB_THREADS_PER_BLOCK
            << std::endl;
    std::cout << "MAX_BLOCKS:                 " << MAX_BLOCKS << std::endl;
    std::cout << "Starting tree depth:        " << start << std::endl;
    std::cout << "Finishing tree depth:       " << end << std::endl;
    std::cout << "Max points per leaf:        " << max_per_leaf << std::endl;
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
        float3 top = make_float3(1., 1., 1.);
        float3 bot = make_float3(0., 0., 0.);


        /* Build the tree from and time it for N_iter iterations. */

        cudaEvent_t part_start, part_stop;
        cudaEvent_t tot_start, tot_stop;
        float part_elapsed;
        double all_tot, morton_tot, sort_tot, tree_tot, compact_tot, aabb_tot;
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

            grace::Tree d_tree(N);

            cudaEventRecord(part_start);
            grace::build_tree(d_tree, d_keys, max_per_leaf);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            tree_tot += part_elapsed;

            cudaEventRecord(part_start);
            grace::compact_tree(d_tree);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            compact_tot += part_elapsed;

            cudaEventRecord(part_start);
            grace::find_AABBs(d_tree, d_spheres_xyzr);
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
        std::cout << "    i)  A tree from " << N << " random points."
                  << std::endl;
        std::cout << "    ii) AABBs for a fully-balanced tree with " << levels
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

        std::cout << "Time for hierarchy compaction:    ";
        std::cout.width(7);
        std::cout << compact_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for calculating AABBs:       ";
        std::cout.width(7);
        std::cout << aabb_tot/N_iter << " ms." << std::endl;

        std::cout << "Time for total (inc. memory ops): ";
        std::cout.width(7);
        std::cout << all_tot/N_iter << " ms." << std::endl;


        /* Build fully-balanced tree on host. */

        grace::H_Tree h_tree(N);

        // Set up bottom level (where all nodes connect to leaves).
        for (unsigned int i_left=1; i_left<N-1; i_left+=4)
        {
            unsigned int i_right = i_left + 1;

            h_tree.nodes[4*i_left].x = i_left - 1 + N-1;
            h_tree.nodes[4*i_left].y = i_left + N-1;
            h_tree.nodes[4*i_left].w = i_left - 1;

            h_tree.nodes[4*i_right].x = i_right + N-1;
            h_tree.nodes[4*i_right].y = i_right + 1 + N-1;
            h_tree.nodes[4*i_right].w = i_right + 1;

            h_tree.leaves[i_left-1].x = i_left-1;
            h_tree.leaves[i_left-1].y = 1;
            h_tree.leaves[i_left-1].z = i_left;

            h_tree.leaves[i_left].x = i_left;
            h_tree.leaves[i_left].y = 1;
            h_tree.leaves[i_left].z = i_left;

            h_tree.leaves[i_right].x = i_right;
            h_tree.leaves[i_right].y = 1;
            h_tree.leaves[i_right].z = i_right;

            h_tree.leaves[i_right+1].x = i_right+1;
            h_tree.leaves[i_right+1].y = 1;
            h_tree.leaves[i_right+1].z = i_right;
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

                h_tree.nodes[4*i_left].x = i_left_split;
                h_tree.nodes[4*i_left].y = i_left_split + 1;
                h_tree.nodes[4*i_left].w = i_left - (1u<<height) + 1;

                h_tree.nodes[4*i_right].x = i_right_split;
                h_tree.nodes[4*i_right].y = i_right_split + 1;
                h_tree.nodes[4*i_right].w = i_right + (1u<<height) - 1;

                h_tree.nodes[4*i_left_split].z =
                    h_tree.nodes[4*(i_left_split+1)].z = i_left;
                h_tree.nodes[4*i_right_split].z =
                    h_tree.nodes[4*(i_right_split+1)].z = i_right;
            }
        }

        // Set up root node and link children to it.
        h_tree.nodes[0].x = N/2 - 1;
        h_tree.nodes[0].y = N/2;
        h_tree.nodes[4*(N/2 - 1)].z = h_tree.nodes[4*(N/2)].z = 0;
        h_tree.nodes[0].w = N - 1;


        /* Profile the fully-balanced tree. */

        grace::Tree d_tree(N);
        part_elapsed = 0;
        aabb_tot = 0;
        for (int i=0; i<N_iter; i++)
        {
            // NB: Levels and AABBs do not need copying: we don't build them on
            // the host.
            d_tree.nodes = h_tree.nodes;
            thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;

            cudaEventRecord(part_start);
            grace::find_AABBs(d_tree, d_spheres_xyzr);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            aabb_tot += part_elapsed;
        }

        aabb_tot /= N_iter;
        std::cout << "Time for balanced tree AABBs:     ";
        std::cout.width(7);
        std::cout << aabb_tot << " ms." << std::endl;
        std::cout << std::endl;
    }

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
