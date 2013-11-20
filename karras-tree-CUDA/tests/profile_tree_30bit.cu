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
#include "../kernels/bintree_build_kernels.cuh"

int main(int argc, char* argv[]) {

    typedef grace::Vector3<float> Vector3f;
    cudaDeviceProp deviceProp;
    std::ofstream outfile;
    std::string file_name;
    std::ostringstream converter;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(5);
    outfile.fill('0');


    /* Initialize run parameters. */

    unsigned int levels = 20;
    unsigned int N_iter = 1000;
    unsigned int file_num = 1;
    unsigned int device_ID = 0;
    unsigned int seed_factor = 1u;
    if (argc > 5) {
        seed_factor = (unsigned int) std::strtol(argv[5], NULL, 10);
    }
    if (argc > 4) {
        levels = (unsigned int) std::strtol(argv[4], NULL, 10);
        // Keep levels in [5, 25].
        levels = min(25, max(5, levels));
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

    unsigned int N = 1u << (levels - 1);
    std::cout << "Will profile (on device " << device_ID << " with " << N_iter
              << " iterations):" << std::endl;
    std::cout << "    i)  A tree constructed from " << N
              << " uniform random positions." << std::endl;
    std::cout << "    ii) AABB finding (only) of a fully balanced tree with "
              << N << " leaves." << std::endl;
    std::cout << std::endl;
    std::cout << "Saving results to " << file_name << std::endl;


    /* Write run parameters to file. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);
    // Wipe the file, if it exists.
    outfile.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);
    outfile << "Device " << device_ID
                    << ":                 " << deviceProp.name << std::endl;
    outfile << "Tree depth:               " << levels << std::endl;
    outfile << "Number of leaves:         " << N << std::endl;
    outfile << "Number of nodes + leaves: " << 2*N - 1 << std::endl;
    outfile << "Iterations per tree:      " << N_iter << std::endl;
    outfile << "Random points' seed factor: " << seed_factor << std::endl;
    outfile << "MORTON_THREADS_PER_BLOCK: " << MORTON_THREADS_PER_BLOCK
            << std::endl;
    outfile << "BUILD_THREADS_PER_BLOCK:  " << BUILD_THREADS_PER_BLOCK
            << std::endl;
    outfile << "AABB_THREADS_PER_BLOCK:   " << AABB_THREADS_PER_BLOCK
            << std::endl;
    outfile << "MAX_BLOCKS:               " << MAX_BLOCKS << std::endl;
    outfile << std::endl << std::endl;
    outfile.close();


    /* Allocate arrays and generate input data. */

    // Generate N random positions and radii, i.e. 4N random floats in [0,1).
    thrust::host_vector<float> h_x_centres(N);
    thrust::host_vector<float> h_y_centres(N);
    thrust::host_vector<float> h_z_centres(N);
    thrust::host_vector<float> h_radii(N);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_x_centres.begin(),
                      random_float_functor(0u, seed_factor) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_y_centres.begin(),
                      random_float_functor(1u, seed_factor) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_z_centres.begin(),
                      random_float_functor(2u, seed_factor) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_radii.begin(),
                      random_float_functor(0.1f, seed_factor) );

    // Set the AABBs.
    Vector3f bottom(0., 0., 0.);
    Vector3f top(1., 1., 1.);


    /* Profile the tree constructed from random data. */

    cudaEvent_t part_start, part_stop;
    cudaEvent_t tot_start, tot_stop;
    float part_elapsed, tot_elapsed;
    float times[5];
    cudaEventCreate(&part_start);
    cudaEventCreate(&part_stop);
    cudaEventCreate(&tot_start);
    cudaEventCreate(&tot_stop);

    for (int i=0; i<N_iter; i++) {
        cudaEventRecord(tot_start);
        // Copy pristine host-side data to GPU.
        thrust::device_vector<float> d_x_centres = h_x_centres;
        thrust::device_vector<float> d_y_centres = h_y_centres;
        thrust::device_vector<float> d_z_centres = h_z_centres;
        thrust::device_vector<float> d_radii = h_radii;

        // Generate the Morton keys for each position.
        thrust::device_vector<UInteger32> d_keys(N);
        cudaEventRecord(part_start);
        grace::morton_keys(d_x_centres, d_y_centres, d_z_centres,
                           d_keys, bottom, top);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        times[1] += part_elapsed;

        // Sort the positions by their keys and save the sorted keys.
        thrust::device_vector<int> d_indices(N);
        thrust::device_vector<float> d_tmp(N);
        cudaEventRecord(part_start);
        thrust::sequence(d_indices.begin(), d_indices.end());
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

        thrust::gather(d_indices.begin(),
                       d_indices.end(),
                       d_x_centres.begin(),
                       d_tmp.begin());
        d_x_centres = d_tmp;

        thrust::gather(d_indices.begin(),
                       d_indices.end(),
                       d_y_centres.begin(),
                       d_tmp.begin());
        d_y_centres = d_tmp;

        thrust::gather(d_indices.begin(),
                       d_indices.end(),
                       d_z_centres.begin(),
                       d_tmp.begin());
        d_z_centres = d_tmp;

        thrust::gather(d_indices.begin(),
                       d_indices.end(),
                       d_radii.begin(),
                       d_tmp.begin());
        d_radii = d_tmp;
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        times[2] += part_elapsed;


        // Build the tree hierarchy from the keys.
        thrust::device_vector<grace::Node> d_nodes(N-1);
        thrust::device_vector<grace::Leaf> d_leaves(N);
        cudaEventRecord(part_start);
        grace::build_nodes(d_nodes, d_leaves, d_keys);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        times[3] += part_elapsed;

        // Find the AABBs.
        cudaEventRecord(part_start);
        grace::find_AABBs(d_nodes, d_leaves,
                          d_x_centres, d_y_centres, d_z_centres, d_radii);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        times[4] += part_elapsed;

        // Record the total time spent in the loop.
        cudaEventRecord(tot_stop);
        cudaEventSynchronize(tot_stop);
        cudaEventElapsedTime(&tot_elapsed, tot_start, tot_stop);
        times[0] += tot_elapsed;
    }


    /* Calculate mean times and write results to file. */

    for (int i=0; i<5; i++) {
        times[i] /= N_iter;
    }

    outfile.open(file_name.c_str(), std::ofstream::out | std::ofstream::app);
    outfile << "Time for morton key generation: " << times[1] << " ms."
            << std::endl;
    outfile << "Time for sort-by-key:           " << times[2] << " ms."
            << std::endl;
    outfile << "Time for hierarchy generation:  " << times[3] << " ms."
            << std::endl;
    outfile << "Time for calculating AABBs:     " << times[4] << " ms."
            << std::endl;
    outfile << "Total time for loop:            " << times[0] << " ms."
            << std::endl;
    outfile << std::endl << std::endl;
    outfile.close();
}
