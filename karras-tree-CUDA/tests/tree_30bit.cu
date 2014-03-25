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

    std::ofstream outfile;
    thrust::host_vector<float> h_write_f;
    thrust::host_vector<float4> h_write_f4;
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


    /* Generate N random points as floats in [0,1) and radii in [0,0.1). */

    thrust::device_vector<float4> d_spheres_xyzr(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres_xyzr.begin(),
                      grace::random_float4_functor(0.1f, seed_factor) );


    /* Save randomly generated data if requested. */

    if (save_in) {
        h_write_f4 = d_spheres_xyzr;
        outfile.open("indata/x_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f4[i].x << std::endl;
        }
        outfile.close();

        outfile.open("indata/y_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f4[i].y << std::endl;
        }
        outfile.close();

        outfile.open("indata/z_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f4[i].z << std::endl;
        }
        outfile.close();

        outfile.open("indata/r_fdata.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << h_write_f4[i].w << std::endl;
        }
        outfile.close();
    }


    /* Generate the Morton key of each position and save them, unsorted. */

    thrust::device_vector<grace::uinteger32> d_keys(N);

    float4 bottom = make_float4(0., 0., 0., 0.);
    float4 top = make_float4(1., 1., 1., 0.);

    grace::morton_keys(d_spheres_xyzr, d_keys, bottom, top);

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

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres_xyzr.begin());

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

    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);

    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);


    /* Save node and leaf data. */

    grace::H_Nodes h_nodes(N-1);
    grace::H_Leaves h_leaves(N);

    h_nodes.hierarchy = d_nodes.hierarchy;
    h_nodes.level = d_nodes.level;
    h_nodes.AABB = d_nodes.AABB;

    h_leaves.parent = d_leaves.parent;
    h_leaves.AABB = d_leaves.AABB;

    if (save_out) {
        outfile.open("outdata/nodes.txt");
        for (unsigned int i=0; i<N-1; i++) {
            outfile << "i:               " << i << std::endl;
            outfile << "level:           " << h_nodes.level[i] << std::endl;
            int4 node = h_nodes.hierarchy[i];
            // Output the actual index into the leaf array for comparison
            // to the Python code.
            if (node.x > N-2) {
                outfile << "left leaf flag:  True" << std::endl;
                outfile << "left:            " << node.x - (N-1) << std::endl;
            }
            else {
                outfile << "left leaf flag:  False" << std::endl;
                outfile << "left:            " << node.x << std::endl;
            }
            if (node.y > N-2) {
                outfile << "right leaf flag:  True" << std::endl;
                outfile << "right:            " << node.y - (N-1) << std::endl;
            }
            else {
                outfile << "right leaf flag:  False" << std::endl;
                outfile << "right:            " << node.y << std::endl;
            }
            outfile << "parent:          " << node.z << std::endl;
            outfile << "AABB_bottom:     " << h_nodes.AABB[i].bx << ", "
                                           << h_nodes.AABB[i].by << ", "
                                           << h_nodes.AABB[i].bz
                                           << std::endl;
            outfile << "AABB_top:        " << h_nodes.AABB[i].tx << ", "
                                           << h_nodes.AABB[i].ty << ", "
                                           << h_nodes.AABB[i].tz << std::endl;
            outfile << std::endl;
        }
        outfile.close();

        outfile.open("outdata/leaves.txt");
        for (unsigned int i=0; i<N; i++) {
            outfile << "i:           " << i << std::endl;
            outfile << "parent:      " << h_leaves.parent[i] << std::endl;
            outfile << "AABB_bottom: " << h_leaves.AABB[i].bx << ", "
                                       << h_leaves.AABB[i].by << ", "
                                       << h_leaves.AABB[i].bz << std::endl;
            outfile << "AABB_top:    " << h_leaves.AABB[i].tx << ", "
                                       << h_leaves.AABB[i].ty << ", "
                                       << h_leaves.AABB[i].tz << std::endl;
            outfile << std::endl;
        }
    }
}
