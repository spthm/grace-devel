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

#define N_TIMES 5

int main(int argc, char* argv[]) {

    typedef grace::Vector3<float> Vector3f;
    cudaDeviceProp deviceProp;
    std::ofstream outfile;
    std::ifstream infile;
    std::string infile_name, outfile_name;
    std::ostringstream converter;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(5);


    /* Initialize run parameters. */

    unsigned int N_iter = 100;
    unsigned int file_num = 1;
    unsigned int device_ID = 0;
    infile_name = "Data_025";
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
    outfile_name = ("profile_tree_gadget_" + converter.str() + ".log");

    std::cout << "Will profile (on device " << device_ID << " with " << N_iter
              << " iterations) a tree from Gadget file " << infile_name
              << std::endl;
    std::cout << std::endl;
    std::cout << "Saving results to " << outfile_name << std::endl;
    std::cout << std::endl;

    infile.open(infile_name.c_str(), std::ios::binary);
    gadget_header header = read_gadget_header(infile);
    infile.close();


    /* Write run parameters to file. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);
    // Wipe the file, if it exists.
    outfile.open(outfile_name.c_str(), std::ofstream::trunc);
    outfile << "Device " << device_ID
                    << ":                   " << deviceProp.name << std::endl;
    outfile << "MORTON_THREADS_PER_BLOCK:   " << MORTON_THREADS_PER_BLOCK
            << std::endl;
    outfile << "BUILD_THREADS_PER_BLOCK:    " << BUILD_THREADS_PER_BLOCK
            << std::endl;
    outfile << "AABB_THREADS_PER_BLOCK:     " << AABB_THREADS_PER_BLOCK
            << std::endl;
    outfile << "MAX_BLOCKS:                 " << MAX_BLOCKS << std::endl;
    outfile << "Iterations per tree:        " << N_iter << std::endl;
    outfile << "Gadget data file name:      " << infile_name << std::endl;
    outfile << std::endl << std::endl;
    outfile.close();


    std::cout << "Profiling tree from Gadget data."<< std::endl;
    unsigned int N = header.npart[0];
    std::cout << "Gadget files contains " << N << " gas particles and "
              << header.npart[1] << " dark matter particles." << std::endl;

    // Generate N random positions and radii, i.e. 4N random floats in [0,1).
    thrust::host_vector<float> h_x_centres(N);
    thrust::host_vector<float> h_y_centres(N);
    thrust::host_vector<float> h_z_centres(N);
    thrust::host_vector<float> h_radii(N);
    thrust::host_vector<float> h_masses(N);
    thrust::host_vector<float> h_rho(N);

    std::cout << "Reading in file..." << std::endl;
    infile.open(infile_name.c_str(), std::ios::binary);
    read_gadget_gas(infile, h_x_centres, h_y_centres, h_z_centres,
                            h_radii, h_masses, h_rho);
    infile.close();
    // Masses unused.  Free space.
    h_masses.clear(); h_masses.shrink_to_fit();
    h_rho.clear(); h_rho.shrink_to_fit();


    // Set the AABBs.
    float max_x = thrust::reduce(h_x_centres.begin(),
                                 h_x_centres.end(),
                                 -1.0f,
                                 thrust::maximum<float>());
    float max_y = thrust::reduce(h_y_centres.begin(),
                                 h_y_centres.end(),
                                 -1.0f,
                                 thrust::maximum<float>());
    float max_z = thrust::reduce(h_z_centres.begin(),
                                 h_z_centres.end(),
                                 -1.0f,
                                 thrust::maximum<float>());
    float min_x = thrust::reduce(h_x_centres.begin(),
                                 h_x_centres.end(),
                                 max_x,
                                 thrust::minimum<float>());
    float min_y = thrust::reduce(h_y_centres.begin(),
                                 h_y_centres.end(),
                                 max_y,
                                 thrust::minimum<float>());
    float min_z = thrust::reduce(h_z_centres.begin(),
                                 h_z_centres.end(),
                                 max_z,
                                 thrust::minimum<float>());
    Vector3f bottom(min_x, min_y, min_z);
    Vector3f top(max_x, max_y, max_z);


    /* Profile the tree constructed from random data. */

    cudaEvent_t part_start, part_stop;
    cudaEvent_t tot_start, tot_stop;
    float part_elapsed, tot_elapsed;
    double times[N_TIMES];
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
        thrust::device_vector<grace::uinteger32> d_keys(N);
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

    // Calculate mean timings and write results to file.
    for (int i=0; i<N_TIMES; i++) {
        times[i] /= N_iter;
    }
    outfile.open(outfile_name.c_str(),
                 std::ofstream::out | std::ofstream::app);
    outfile << "Will generate:" << std::endl;
    outfile << "    i)  A tree from " << N << " SPH particles." << std::endl;
    outfile << std::endl;
    outfile << "Time for Morton key generation:    ";
    outfile.width(8);
    outfile << times[1] << " ms." << std::endl;
    outfile << "Time for sort-by-key:              ";
    outfile.width(8);
    outfile << times[2] << " ms." << std::endl;
    outfile << "Time for hierarchy generation:     ";
    outfile.width(8);
    outfile << times[3] << " ms." << std::endl;
    outfile << "Time for calculating AABBs:        ";
    outfile.width(8);
    outfile << times[4] << " ms." << std::endl;
    outfile << "Time for total (inc. memory ops): ";
    outfile.width(8);
    outfile << times[0] << " ms." << std::endl;
    outfile << std::endl << std::endl;
    outfile.close();
}
