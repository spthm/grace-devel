#include <cstring>
#include <fstream>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "utils.cuh"
#include "../types.h"
#include "../nodes.h"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;
    std::ofstream outfile;
    std::string file_name;
    std::ostringstream converter;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(5);


    /* Initialize run parameters. */

    unsigned int start = 20;
    unsigned int end = 20;
    unsigned int N_iter = 100;
    unsigned int file_num = 1;
    unsigned int device_ID = 0;
    unsigned int seed_factor = 1u;
    if (argc > 6) {
        seed_factor = (unsigned int) std::strtol(argv[5], NULL, 10);
    }
    if (argc > 5) {
        end = (unsigned int) std::strtol(argv[5], NULL, 10);
        // Keep levels in [5, 28].
        end = min(28, max(5, end));
        start = (unsigned int) std::strtol(argv[4], NULL, 10);
        start = min(28, max(5, start));
    }
    else if (argc > 4) {
        end = (unsigned int) std::strtol(argv[4], NULL, 10);
        end = min(28, max(5, end));
        if (end < start)
            start = end;
    }
    if (argc > 3) {
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
    }
    if (argc > 2) {
        device_ID = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 1) {
        file_num = (unsigned int) std::strtol(argv[1], NULL, 10);

    }
    // Convert file number to a string.
    converter << file_num;
    file_name = ("profile_tree_" + converter.str() + ".log");

    std::cout << "Will profile (on device " << device_ID << " with " << N_iter
              << " iterations):" << std::endl;
    std::string N_leaves_str;
    if (start == end) {
        converter.seekp(0);
        converter << (1u << (start-1));
        N_leaves_str = converter.str();
    }
    else {
        N_leaves_str = std::string("2^levels - 1");
    }
    std::cout << "    i)  A tree constructed from "
              << N_leaves_str << " uniform random positions." << std::endl;
    std::cout << "    ii) AABB finding (only) of a fully balanced tree with "
              << N_leaves_str << " leaves." << std::endl;
    if (start != end)
        std::cout << "For levels = " << start << " to " << end << std::endl;
    std::cout << std::endl;
    std::cout << "Saving results to " << file_name << std::endl;
    std::cout << std::endl;


    /* Write run parameters to file. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);
    // Wipe the file, if it exists.
    outfile.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);
    outfile << "Device " << device_ID
                    << ":                   " << deviceProp.name << std::endl;
    outfile << "MORTON_THREADS_PER_BLOCK:   " << MORTON_THREADS_PER_BLOCK
            << std::endl;
    outfile << "BUILD_THREADS_PER_BLOCK:    " << BUILD_THREADS_PER_BLOCK
            << std::endl;
    outfile << "AABB_THREADS_PER_BLOCK:     " << AABB_THREADS_PER_BLOCK
            << std::endl;
    outfile << "MAX_BLOCKS:                 " << MAX_BLOCKS << std::endl;
    outfile << "Starting tree depth:        " << start << std::endl;
    outfile << "Finishing tree depth:       " << end << std::endl;
    outfile << "Iterations per tree:        " << N_iter << std::endl;
    outfile << "Random points' seed factor: " << seed_factor << std::endl;
    outfile << std::endl << std::endl;
    outfile.close();


    for (int levels=start; levels<=end; levels++)
    {
        std::cout << "Profiling tree of depth " << levels << "..." << std::endl;
        unsigned int N = 1u << (levels - 1);
        /*******************************************************************/
        /* Allocate vectors and generate input data for random-point tree. */
        /*******************************************************************/

        // Generate N random positions and radii, i.e. 4N random floats in [0,1).
        thrust::host_vector<float4> h_spheres_xyzr(N);
        thrust::transform(thrust::counting_iterator<unsigned int>(0),
                          thrust::counting_iterator<unsigned int>(N),
                          h_spheres_xyzr.begin(),
                          grace::random_float4_functor(0.1f, seed_factor) );

        // Set the tree AABB.
        float4 bottom = make_float4(0., 0., 0., 0.);
        float4 top = make_float4(1., 1., 1., 0.);


        /* Profile the tree constructed from random data. */

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

            // Generate the Morton keys for each position.
            thrust::device_vector<grace::uinteger32> d_keys(N);
            cudaEventRecord(part_start);
            grace::morton_keys(d_spheres_xyzr, d_keys, bottom, top);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            morton_tot += part_elapsed;

            // Sort the positions by their keys.
            cudaEventRecord(part_start);
            thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                                d_spheres_xyzr.begin());
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            sort_tot += part_elapsed;


            // Build the tree hierarchy from the keys.
            grace::Nodes d_nodes(N-1);
            grace::Leaves d_leaves(N);
            cudaEventRecord(part_start);
            grace::build_nodes(d_nodes, d_leaves, d_keys);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            tree_tot += part_elapsed;

            // Find the AABBs.
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

        outfile.open(file_name.c_str(), std::ofstream::out | std::ofstream::app);
        outfile << "Will generate:" << std::endl;
        outfile << "    i)  A tree from " << N << " random points." << std::endl;
        outfile << "    ii) A fully-balanced tree with " << levels
                << " levels and " << N << " leaves." << std::endl;
        outfile << std::endl;
        outfile << "Time for Morton key generation:    ";
        outfile.width(8);
        outfile << morton_tot/N_iter << " ms." << std::endl;
        outfile << "Time for sort-by-key:              ";
        outfile.width(8);
        outfile << sort_tot/N_iter << " ms." << std::endl;
        outfile << "Time for hierarchy generation:     ";
        outfile.width(8);
        outfile << tree_tot/N_iter << " ms." << std::endl;
        outfile << "Time for calculating AABBs:        ";
        outfile.width(8);
        outfile << aabb_tot/N_iter << " ms." << std::endl;
        outfile << "Time for total (inc. memory ops):  ";
        outfile.width(8);
        outfile << all_tot/N_iter << " ms." << std::endl;
        outfile.close();



        /**************************************/
        /* Build fully-balanced tree on host. */
        /**************************************/

        // Alloctate host-side vectors.
        grace::H_Nodes h_nodes(N-1);
        grace::H_Leaves h_leaves(N);

        // Set up bottom level (where all nodes connect to leaves).
        for (unsigned int i_left=1; i_left<N-1; i_left+=4)
        {
            unsigned int i_right = i_left + 1;

            h_nodes.hierarchy[i_left].x = i_left - 1 + N-1; // Left.
            h_nodes.hierarchy[i_left].y = i_left + N-1; // Right.
            h_nodes.hierarchy[i_left].w = i_left - 1; // End.

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

                h_nodes.hierarchy[i_left_split].z = h_nodes.hierarchy[i_left_split+1].z
                                             = i_left;
                h_nodes.hierarchy[i_right_split].z = h_nodes.hierarchy[i_right_split+1].z
                                              = i_right;
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
            // NB: Levels and AABBs do not need copying since we don't
            // build them on the host.
            d_nodes.hierarchy = h_nodes.hierarchy;
            thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;

            // Find the AABBs.
            cudaEventRecord(part_start);
            grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
            aabb_tot += part_elapsed;
        }

        // Calculate mean timings and write results to file.
        aabb_tot /= N_iter;
        outfile.open(file_name.c_str(), std::ofstream::out | std::ofstream::app);
        outfile << "Time for balanced tree AABBs:      ";
        outfile.width(8);
        outfile << aabb_tot << " ms." << std::endl;
        outfile << std::endl << std::endl;
        outfile.close();
    }
}
