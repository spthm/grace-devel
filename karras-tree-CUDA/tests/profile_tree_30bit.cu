#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <fstream>

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

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(5);
    outfile.fill('0');


    /* Generate N random positions, i.e. 3*N random floats in [0,1) */

    unsigned int levels = 20;
    unsigned int seed_factor = 1u;
    if (argc > 2) {
        seed_factor = (unsigned int) std::strtol(argv[3], NULL, 10);
    }
    else if (argc > 1) {
        levels = (unsigned int) std::strtol(argv[1], NULL, 10);
        // Keep levels in [5, 25].
        levels = min(25, max(5, levels));
    }
    unsigned int N = 1u << (levels - 1);
    std::cout << "Will profile a tree with " << N << " uniform random points."
              << std::endl;
    std::cout << "Will profile a fully balanced tree with " << N << " leaves."
              << std::endl;

    thrust::device_vector<float> d_x_centres(N);
    thrust::device_vector<float> d_y_centres(N);
    thrust::device_vector<float> d_z_centres(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_x_centres.begin(),
                      random_float_functor(0u, seed_factor) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_y_centres.begin(),
                      random_float_functor(1u, seed_factor) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_z_centres.begin(),
                      random_float_functor(2u, seed_factor) );


    /* Generate N random radii as floats in [0,1). */

    thrust::device_vector<float> d_radii(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_radii.begin(),
                      random_float_functor(0.1f, seed_factor) );


    /* Save randomly generated data if requested. */

    if (save_in) {
        h_write_f = d_x_centres;
        outfile.open("indata/x_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f[i] << std::endl;
        }
        outfile.close();

        h_write_f = d_y_centres;
        outfile.open("indata/y_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f[i] << std::endl;
        }
        outfile.close();

        h_write_f = d_z_centres;
        outfile.open("indata/z_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f[i] << std::endl;
        }
        outfile.close();

        h_write_f = d_radii;
        outfile.open("indata/r_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f[i] << std::endl;
        }
        outfile.close();
    }


    /* Generate the Morton key of each position and save them, unsorted. */

    thrust::device_vector<UInteger32> d_keys(N);

    Vector3f bottom(0., 0., 0.);
    Vector3f top(1., 1., 1.);

    grace::morton_keys(d_x_centres, d_y_centres, d_z_centres,
                      d_keys, bottom, top);

    if (save_out) {
        h_write_uint = d_keys;
        outfile.open("outdata/unsorted_keys_base10.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_uint[i] << std::endl;
        }
        outfile.close();

        outfile.open("outdata/unsorted_keys_base2.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << (std::bitset<32>) h_write_uint[i] << std::endl;
        }
        outfile.close();
    }


    /* Sort the position vectors by their keys and save sorted keys. */

    thrust::device_vector<int> d_indices(N);
    thrust::device_vector<float> d_tmp(N);
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

    if (save_out) {
        h_write_uint = d_keys;
        outfile.open("outdata/sorted_keys_base10.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_uint[i] << std::endl;
        }
        outfile.close();

        outfile.open("outdata/sorted_keys_base2.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << (std::bitset<32>) h_write_uint[i] << std::endl;
        }
        outfile.close();
    }


    /* Build the tree from the keys. */

    thrust::device_vector<grace::Node> d_nodes(N-1);
    thrust::device_vector<grace::Leaf> d_leaves(N);

    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves,
                      d_x_centres, d_y_centres, d_z_centres, d_radii);


    /* Save node and leaf data. */

    thrust::host_vector<grace::Node> h_nodes = d_nodes;
    thrust::host_vector<grace::Leaf> h_leaves = d_leaves;

    if (save_out) {
        outfile.open("outdata/nodes.txt");
        for (unsigned int i=0; i<N-1; i++) {
            outfile << "i:               " << i << std::endl;
            outfile << "level:           " << h_nodes[i].level << std::endl;
            outfile << "left leaf flag:  "
                    << (h_nodes[i].left_leaf_flag ? "True" : "False") << std::endl;
            outfile << "left:            " << h_nodes[i].left << std::endl;
            outfile << "right leaf flag: "
                    << (h_nodes[i].right_leaf_flag ? "True": "False")<< std::endl;
            outfile << "right:           " << h_nodes[i].right << std::endl;
            outfile << "parent:          " << h_nodes[i].parent << std::endl;
            outfile << "AABB_bottom:     " << h_nodes[i].bottom[0] << ", "
                                           << h_nodes[i].bottom[1] << ", "
                                           << h_nodes[i].bottom[2] << std::endl;
            outfile << "AABB_top:        " << h_nodes[i].top[0] << ", "
                                           << h_nodes[i].top[1] << ", "
                                           << h_nodes[i].top[2] << std::endl;
            outfile << std::endl;
        }
        outfile.close();

        outfile.open("outdata/leaves.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << "i:           " << i << std::endl;
            outfile << "parent:      " << h_leaves[i].parent << std::endl;
            outfile << "AABB_bottom: " << h_leaves[i].bottom[0] << ", "
                                       << h_leaves[i].bottom[1] << ", "
                                       << h_leaves[i].bottom[2] << std::endl;
            outfile << "AABB_top:    " << h_leaves[i].top[0] << ", "
                                       << h_leaves[i].top[1] << ", "
                                       << h_leaves[i].top[2] << std::endl;
            outfile << std::endl;
        }
    }
}
