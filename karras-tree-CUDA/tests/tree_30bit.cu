#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../utils.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"

int main(int argc, char* argv[]) {

    std::ofstream outfile;
    thrust::host_vector<float> h_write_f;
    thrust::host_vector<float4> h_write_f4;
    thrust::host_vector<grace::uinteger32> h_write_uint;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(9);


    /* Initialize run parameters. */

    unsigned int N = 100000;
    unsigned int max_per_leaf = 1;
    bool save_in = false;
    bool save_out = false;
    unsigned int seed_factor = 1u;

    if (argc > 1) {
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        max_per_leaf = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        if (strcmp("in", argv[3]) == 0)
            save_in = true;
        else if (strcmp("out", argv[3]) == 0)
            save_out = true;
        else if (strcmp("inout", argv[3]) == 0)
            save_in = save_out = true;
    }
    if (argc > 4) {
        seed_factor = (unsigned int) std::strtol(argv[4], NULL, 10);
    }

    std::cout << "Will generate " << N << " random points, with up to "
              << max_per_leaf << " point(s) per leaf." << std::endl;
    if (save_in == save_out) {
        if (save_in)
            std::cout << "Will save all data." << std::endl;
    }
    else {
        if (save_in)
            std::cout << "Will save random floating point data." << std::endl;
        else
            std::cout << "Will save key, node and leaf data." << std::endl;
    }

{ // Device code.


    /* Generate N random points as floats in [0,1) and radii in [0,0.1). */

    thrust::device_vector<float4> d_spheres_xyzr(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres_xyzr.begin(),
                      grace::random_float4_functor(0.1f, seed_factor));


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


    /* Generate the Morton key of each (unsorted) sphere and save it. */

    thrust::device_vector<grace::uinteger32> d_keys(N);

    float3 top = make_float3(1., 1., 1.);
    float3 bot = make_float3(0., 0., 0.);

    grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);

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


    /* Sort the spheres by their keys and save the sorted keys. */

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

    grace::build_nodes(d_nodes, d_leaves, d_keys, max_per_leaf);
    grace::compact_nodes(d_nodes, d_leaves);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);


    /* Save node and leaf data. */

    grace::H_Nodes h_nodes(N-1);
    grace::H_Leaves h_leaves(N);

    h_nodes.hierarchy = d_nodes.hierarchy;
    h_nodes.level = d_nodes.level;
    h_nodes.AABB = d_nodes.AABB;

    h_leaves.indices = d_leaves.indices;
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
                outfile << "right leaf flag: True" << std::endl;
                outfile << "right:           " << node.y - (N-1) << std::endl;
            }
            else {
                outfile << "right leaf flag: False" << std::endl;
                outfile << "right:           " << node.y << std::endl;
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
        outfile << "Max spheres per leaf: " << max_per_leaf << std::endl;
        outfile << std::endl;
        for (unsigned int i=0; i<N; i++) {
            int4 leaf = h_leaves.indices[i];
            outfile << "i:            " << i << std::endl;
            outfile << "first sphere: " << leaf.x << std::endl;
            outfile << "sphere count: " << leaf.y << std::endl;
            outfile << "parent:       " << leaf.z << std::endl;
            outfile << "AABB_bottom:  " << h_leaves.AABB[i].bx << ", "
                                        << h_leaves.AABB[i].by << ", "
                                        << h_leaves.AABB[i].bz << std::endl;
            outfile << "AABB_top:     " << h_leaves.AABB[i].tx << ", "
                                        << h_leaves.AABB[i].ty << ", "
                                        << h_leaves.AABB[i].tz << std::endl;
            outfile << std::endl;
        }
    }
} // End device code.

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
