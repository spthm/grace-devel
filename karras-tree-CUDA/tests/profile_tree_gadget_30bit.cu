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
    std::ifstream infile;
    std::string infile_name, outfile_name;
    std::ostringstream converter;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(5);


    /* Initialize run parameters. */

    unsigned int file_num = 1;
    unsigned int device_ID = 0;
    unsigned int N_iter = 100;
    infile_name = "Data_025";

    if (argc > 1) {
        file_num = (unsigned int) std::strtol(argv[1], NULL, 10);

    }
    if (argc > 2) {
        device_ID = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
    }


    // Converts file number to a string.
    converter << file_num;
    outfile_name = ("profile_tree_gadget_" + converter.str() + ".log");

    std::cout << "Will profile (on device " << device_ID << " with " << N_iter
              << " iterations) a tree from Gadget file " << infile_name
              << std::endl;
    std::cout << std::endl;
    std::cout << "Saving results to " << outfile_name << std::endl;
    std::cout << std::endl;

    infile.open(infile_name.c_str(), std::ios::binary);
    grace::gadget_header header = grace::read_gadget_header(infile);
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


    /* Read in Gadget data. */

    thrust::host_vector<float4> h_spheres_xyzr(N);
    thrust::host_vector<float> h_masses(N);
    thrust::host_vector<float> h_rho(N);

    std::cout << "Reading in file..." << std::endl;
    infile.open(infile_name.c_str(), std::ios::binary);
    grace::read_gadget_gas(infile, h_spheres_xyzr, h_masses, h_rho);
    infile.close();
    // Masses unused.  Free space.
    h_masses.clear(); h_masses.shrink_to_fit();
    h_rho.clear(); h_rho.shrink_to_fit();

    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;

    // Set the tree-build AABB (contains all sphere centres).
    float min_x, max_x;
    grace::min_max_x(&min_x, &max_x, d_spheres_xyzr);

    float min_y, max_y;
    grace::min_max_y(&min_y, &max_y, d_spheres_xyzr);

    float min_z, max_z;
    grace::min_max_z(&min_z, &max_z, d_spheres_xyzr);

    float4 bot = make_float4(min_x, min_y, min_z, 0.f);
    float4 top = make_float4(max_x, max_y, max_z, 0.f);


    /* Profile the tree constructed from Gadget data. */

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
        grace::morton_keys(d_spheres_xyzr, d_keys, bot, top);
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
        grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);
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

    outfile.open(outfile_name.c_str(),
                 std::ofstream::out | std::ofstream::app);
    outfile << "Will generate:" << std::endl;
    outfile << "    i)  A tree from " << N << " SPH particles." << std::endl;
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
    outfile << "Time for total (inc. memory ops): ";
    outfile.width(8);
    outfile << all_tot/N_iter << " ms." << std::endl;
    outfile << std::endl << std::endl;
    outfile.close();
}
