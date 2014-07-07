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

    grace::Tree d_tree(N);

    grace::build_tree(d_tree, d_keys, max_per_leaf);
    grace::compact_tree(d_tree);
    grace::find_AABBs(d_tree, d_spheres_xyzr);

    /* Save node and leaf data. */

    size_t N_leaves = d_tree.leaves.size();

    grace::H_Tree h_tree(N_leaves);

    h_tree.nodes = d_tree.nodes;
    h_tree.levels = d_tree.levels;
    h_tree.leaves = d_tree.leaves;

    // Copy AABBs into a more output-friendly format.
    // AABB[2*i+0] = (bx, by, bz) of the ith node.
    // AABB[2*i+1] = (tx, ty, tz) of the ith node.
    thrust::host_vector<float3> node_AABBs(2*(N_leaves-1));
    thrust::host_vector<float3> leaf_AABBs(2*N_leaves);
    for (int i=0; i<N_leaves-1; i++) {
        int left_i = h_tree.nodes[4*i+0].x;
        int right_i = h_tree.nodes[4*i+0].y;

        // Left child's AABB.
        // NB: bot and top defined as float3s above.
        bot.x = *reinterpret_cast<float*>(&h_tree.nodes[4*i+1].x);
        bot.y = *reinterpret_cast<float*>(&h_tree.nodes[4*i+2].x);
        bot.z = *reinterpret_cast<float*>(&h_tree.nodes[4*i+3].x);
        top.x = *reinterpret_cast<float*>(&h_tree.nodes[4*i+1].y);
        top.y = *reinterpret_cast<float*>(&h_tree.nodes[4*i+2].y);
        top.z = *reinterpret_cast<float*>(&h_tree.nodes[4*i+3].y);
        if (left_i < N_leaves-1) {
            // Left is a node.
            node_AABBs[2*left_i+0] = bot;
            node_AABBs[2*left_i+1] = top;
        }
        else {
            // Left is a leaf.
            left_i -= (N_leaves-1);
            leaf_AABBs[2*left_i+0] = bot;
            leaf_AABBs[2*left_i+1] = top;
        }

        // Right child's AABB.
        bot.x = *reinterpret_cast<float*>(&h_tree.nodes[4*i+1].z);
        bot.y = *reinterpret_cast<float*>(&h_tree.nodes[4*i+2].z);
        bot.z = *reinterpret_cast<float*>(&h_tree.nodes[4*i+3].z);
        top.x = *reinterpret_cast<float*>(&h_tree.nodes[4*i+1].w);
        top.y = *reinterpret_cast<float*>(&h_tree.nodes[4*i+2].w);
        top.z = *reinterpret_cast<float*>(&h_tree.nodes[4*i+3].w);
        if (right_i < N_leaves-1) {
            // Right is a node.
            node_AABBs[2*right_i+0] = bot;
            node_AABBs[2*right_i+1] = top;
        }
        else {
            // Right is a leaf.
            right_i -= (N_leaves-1);
            leaf_AABBs[2*right_i+0] = bot;
            leaf_AABBs[2*right_i+1] = top;
        }
    }

    // The root node's AABB is implicit.  Compute it.
    bot.x = min(*reinterpret_cast<float*>(&h_tree.nodes[1].x),
                *reinterpret_cast<float*>(&h_tree.nodes[1].z));
    bot.y = min(*reinterpret_cast<float*>(&h_tree.nodes[2].x),
                *reinterpret_cast<float*>(&h_tree.nodes[2].z));
    bot.z = min(*reinterpret_cast<float*>(&h_tree.nodes[3].x),
                *reinterpret_cast<float*>(&h_tree.nodes[3].z));
    top.x = max(*reinterpret_cast<float*>(&h_tree.nodes[1].y),
                *reinterpret_cast<float*>(&h_tree.nodes[1].w));
    top.y = max(*reinterpret_cast<float*>(&h_tree.nodes[2].y),
                *reinterpret_cast<float*>(&h_tree.nodes[2].w));
    top.z = max(*reinterpret_cast<float*>(&h_tree.nodes[3].y),
                *reinterpret_cast<float*>(&h_tree.nodes[3].w));
    node_AABBs[0] = bot;
    node_AABBs[1] = top;

    if (save_out) {
        outfile.open("outdata/nodes.txt");
        for (unsigned int i=0; i<N_leaves-1; i++) {
            outfile << "i:               " << i << std::endl;
            outfile << "level:           " << h_tree.levels[i] << std::endl;
            int4 node = h_tree.nodes[4*i];
            // Output the actual index into the leaf array for comparison
            // to the Python code.
            if (node.x > N_leaves-2) {
                outfile << "left leaf flag:  True" << std::endl;
                outfile << "left:            " << node.x - (N_leaves-1)
                        << std::endl;
            }
            else {
                outfile << "left leaf flag:  False" << std::endl;
                outfile << "left:            " << node.x << std::endl;
            }
            if (node.y > N_leaves-2) {
                outfile << "right leaf flag: True" << std::endl;
                outfile << "right:           " << node.y - (N_leaves-1)
                        << std::endl;
            }
            else {
                outfile << "right leaf flag: False" << std::endl;
                outfile << "right:           " << node.y << std::endl;
            }
            outfile << "parent:          " << node.z << std::endl;
            outfile << "AABB_bottom:     " << node_AABBs[2*i+0].x << ", "
                                           << node_AABBs[2*i+0].y << ", "
                                           << node_AABBs[2*i+0].z
                                           << std::endl;
            outfile << "AABB_top:        " << node_AABBs[2*i+1].x << ", "
                                           << node_AABBs[2*i+1].y << ", "
                                           << node_AABBs[2*i+1].z << std::endl;
            outfile << std::endl;
        }
        outfile.close();

        outfile.open("outdata/leaves.txt");
        outfile << "Max spheres per leaf: " << max_per_leaf << std::endl;
        outfile << std::endl;
        for (unsigned int i=0; i<N_leaves; i++) {
            int4 leaf = h_tree.leaves[i];
            outfile << "i:            " << i << std::endl;
            outfile << "first sphere: " << leaf.x << std::endl;
            outfile << "sphere count: " << leaf.y << std::endl;
            outfile << "parent:       " << leaf.z << std::endl;
            outfile << "AABB_bottom:  " << leaf_AABBs[2*i+0].x << ", "
                                        << leaf_AABBs[2*i+0].y << ", "
                                        << leaf_AABBs[2*i+0].z << std::endl;
            outfile << "AABB_top:     " << leaf_AABBs[2*i+1].x << ", "
                                        << leaf_AABBs[2*i+1].y << ", "
                                        << leaf_AABBs[2*i+1].z << std::endl;
            outfile << std::endl;
        }
    }
} // End device code.

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
