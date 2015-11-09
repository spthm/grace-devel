#include <cstring>
#include <fstream>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../utils.cuh"
#include "../device/build_functors.cuh"
#include "../kernels/albvh.cuh"
#include "../kernels/build_sph.cuh"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;
    std::ifstream infile;
    std::string infile_name;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);


    /* Initialize run parameters. */

    unsigned int device_ID = 0;
    unsigned int max_per_leaf = 32;
    unsigned int N_iter = 100;
    infile_name = "Data_025";

    if (argc > 1) {
        device_ID = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        max_per_leaf = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
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
    std::cout << "MAX_BLOCKS:                 "
              << grace::MAX_BLOCKS << std::endl;
    std::cout << "Max particles per leaf:     " << max_per_leaf << std::endl;
    std::cout << "Iterations per tree:        " << N_iter << std::endl;
    std::cout << "Gadget data file name:      " << infile_name << std::endl;
    std::cout << std::endl << std::endl;


    /* Read in Gadget data. */

    // Arrays are resized in read_gadget_gas()
    thrust::host_vector<float4> h_spheres_xyzr(1);
    thrust::host_vector<unsigned int> h_gadget_IDs(1);
    thrust::host_vector<float> h_masses(1);
    thrust::host_vector<float> h_rho(1);

    infile.open(infile_name.c_str(), std::ios::binary);
    grace::read_gadget_gas(infile, h_spheres_xyzr,
                                   h_gadget_IDs,
                                   h_masses,
                                   h_rho);
    infile.close();

    size_t N = h_spheres_xyzr.size();

    // Gadget IDs, masses and densities unused.
    h_gadget_IDs.clear(); h_gadget_IDs.shrink_to_fit();
    h_masses.clear(); h_masses.shrink_to_fit();
    h_rho.clear(); h_rho.shrink_to_fit();

    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;


    /* Profile the tree constructed from Gadget data. */

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
        grace::morton_keys_sph(d_spheres_xyzr, d_keys);
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
        grace::euclidean_deltas_sph(d_spheres_xyzr, d_deltas);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        deltas_tot += part_elapsed;

        grace::Tree d_tree(N, max_per_leaf);
        thrust::device_vector<int2> d_tmp_nodes(N - 1);

        cudaEventRecord(part_start);
        grace::ALBVH::build_leaves(
            d_tmp_nodes,
            d_tree.leaves,
            d_tree.max_per_leaf,
            thrust::raw_pointer_cast(d_deltas.data()),
            thrust::less<float>());
        grace::ALBVH::remove_empty_leaves(d_tree);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        leaves_tot += part_elapsed;

        const size_t n_new_leaves = d_tree.leaves.size();
        thrust::device_vector<float> d_new_deltas(n_new_leaves + 1);

        cudaEventRecord(part_start);
        grace::ALBVH::copy_leaf_deltas(
            d_tree.leaves,
            thrust::raw_pointer_cast(d_deltas.data()),
            thrust::raw_pointer_cast(d_new_deltas.data()));
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        leaf_deltas_tot += part_elapsed;

        cudaEventRecord(part_start);
        grace::ALBVH::build_nodes(
            d_tree,
            thrust::raw_pointer_cast(d_spheres_xyzr.data()),
            thrust::raw_pointer_cast(d_new_deltas.data()),
            thrust::less<float>(),
            grace::AABB_sphere());
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&part_elapsed, part_start, part_stop);
        nodes_tot += part_elapsed;

        cudaEventRecord(tot_stop);
        cudaEventSynchronize(tot_stop);
        cudaEventElapsedTime(&part_elapsed, tot_start, tot_stop);
        all_tot += part_elapsed;
    }

    std::cout << "Will generate a tree from " << N << " SPH particles."
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
