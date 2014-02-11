#include <iostream>
#include <iomanip>
#include <bitset>
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
#include "../kernels/bintree_build.cuh"

int main(int argc, char* argv[]) {

    typedef grace::Vector3<float> Vector3f;
    std::ofstream outfile;
    thrust::host_vector<float> h_write_f;
    thrust::host_vector<grace::uinteger32> h_write_uint;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(9);
    outfile.width(11);
    outfile.fill('0');


    /* Generate N random positions, i.e. 3*N random floats in [0,1) */

    unsigned int N = 100000;
    bool save_in = false;
    bool save_out = false;
    unsigned int seed_factor = 1u;
    if (argc > 3) {
        seed_factor = (unsigned int) std::strtol(argv[3], NULL, 10);
    }
    if (argc > 2) {
        if (strcmp("in", argv[2]) == 0) {
            save_in = true;
            std::cout << "Will save random floating point data." << std::endl;
        }
        else if (strcmp("out", argv[2]) == 0) {
            save_out = true;
            std::cout << "Will save key, node and leaf data." << std::endl;
        }
        else if (strcmp("inout", argv[2]) == 0) {
            save_in = true;
            save_out = true;
            std::cout << "Will save all data." << std::endl;
        }
    }
    if (argc > 1) {
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    std::cout << "Will generate " << N << " random points." << std::endl;

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
                      random_float_functor(0.0f, 0.1f, seed_factor) );


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

    thrust::device_vector<grace::uinteger32> d_keys(N);

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

    thrust::device_vector<grace::integer32> d_nodes_left(N-1);
    thrust::device_vector<grace::integer32> d_nodes_right(N-1);
    thrust::device_vector<grace::integer32> d_nodes_parent(N-1);
    thrust::device_vector<grace::integer32> d_nodes_end(N-1);
    thrust::device_vector<unsigned int> d_nodes_level(N-1);
    thrust::device_vector<float> d_nodes_top(3*(N-1));
    thrust::device_vector<float> d_nodes_bottom(3*(N-1));

    thrust::device_vector<grace::integer32> d_leaves_parent(N);
    thrust::device_vector<float> d_leaves_top(3*N);
    thrust::device_vector<float> d_leaves_bottom(3*N);

    grace::build_nodes(d_nodes_left,
                       d_nodes_right,
                       d_nodes_parent,
                       d_nodes_end,
                       d_nodes_level,
                       d_leaves_parent,
                       d_keys);
    grace::find_AABBs(d_nodes_left,
                      d_nodes_right,
                      d_nodes_parent,
                      d_nodes_end,
                      d_nodes_top,
                      d_nodes_bottom,
                      d_leaves_parent,
                      d_leaves_top,
                      d_leaves_bottom,
                      d_x_centres, d_y_centres, d_z_centres, d_radii);


    /* Save node and leaf data. */

    thrust::host_vector<grace::integer32> h_nodes_left = d_nodes_left;
    thrust::host_vector<grace::integer32> h_nodes_right = d_nodes_right;
    thrust::host_vector<grace::integer32> h_nodes_parent = d_nodes_parent;
    thrust::host_vector<grace::integer32> h_nodes_end = d_nodes_end;
    thrust::host_vector<unsigned int> h_nodes_level = d_nodes_level;
    thrust::host_vector<float> h_nodes_top = d_nodes_top;
    thrust::host_vector<float> h_nodes_bottom = d_nodes_bottom;

    thrust::host_vector<grace::integer32> h_leaves_parent = d_leaves_parent;
    thrust::host_vector<float> h_leaves_top = d_leaves_top;
    thrust::host_vector<float> h_leaves_bottom = d_leaves_bottom;

    if (save_out) {
        outfile.open("outdata/nodes.txt");
        for (unsigned int i=0; i<N-1; i++) {
            outfile << "i:               " << i << std::endl;
            outfile << "level:           " << h_nodes_level[i] << std::endl;
            // Output the actual index into the leaf array for comparison
            // to the Python code.
            if (h_nodes_left[i] > N-2) {
                outfile << "left leaf flag:  True" << std::endl;
                outfile << "left:            " << h_nodes_left[i] - (N-1)
                        << std::endl;
            }
            else {
                outfile << "left leaf flag:  False" << std::endl;
                outfile << "left:            " << h_nodes_left[i] << std::endl;
            }
            if (h_nodes_right[i]> N-2) {
                outfile << "right leaf flag:  True" << std::endl;
                outfile << "right:            " << h_nodes_right[i] - (N-1)
                        << std::endl;
            }
            else {
                outfile << "right leaf flag:  False" << std::endl;
                outfile << "right:            " << h_nodes_right[i]
                        << std::endl;
            }
            outfile << "parent:          " << h_nodes_parent[i] << std::endl;
            outfile << "AABB_bottom:     " << h_nodes_bottom[3*i+0] << ", "
                                           << h_nodes_bottom[3*i+1] << ", "
                                           << h_nodes_bottom[3*i+2]
                                           << std::endl;
            outfile << "AABB_top:        " << h_nodes_top[3*i+0] << ", "
                                           << h_nodes_top[3*i+1] << ", "
                                           << h_nodes_top[3*i+2] << std::endl;
            outfile << std::endl;
        }
        outfile.close();

        outfile.open("outdata/leaves.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << "i:           " << i << std::endl;
            outfile << "parent:      " << h_leaves_parent[i] << std::endl;
            outfile << "AABB_bottom: " << h_leaves_bottom[3*i+0] << ", "
                                       << h_leaves_bottom[3*i+1] << ", "
                                       << h_leaves_bottom[3*i+2] << std::endl;
            outfile << "AABB_top:    " << h_leaves_top[3*i+0] << ", "
                                       << h_leaves_top[3*i+1] << ", "
                                       << h_leaves_top[3*i+2] << std::endl;
            outfile << std::endl;
        }
    }
}
