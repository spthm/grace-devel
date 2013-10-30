#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "tree_climb_kernels.cuh"

// Thomas Wang hash.
__host__ __device__ unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

class random_float_functor
{
public:

    __host__ __device__ float operator() (unsigned int n)
    {
        unsigned int seed = hash(n);
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> u01(0,1);

        return u01(rng);
    }
};

void check_flags(const thrust::device_vector<unsigned int>& d_flags,
                 const char kernel_type[])
{
    unsigned int minimum = thrust::reduce(d_flags.begin(),
                                          d_flags.end(),
                                          10u,
                                          thrust::minimum<unsigned int>());
    unsigned int maximum = thrust::reduce(d_flags.begin(),
                                          d_flags.end(),
                                          0u,
                                          thrust::maximum<unsigned int>());
    if (minimum != 2) {
        std::cout << "Error in " << kernel_type << " kernel:" << std::endl;
        std::cout << "Minimum flag: " << minimum << " != 2." << std::endl;
    }
    if (maximum != 2) {
        std::cout << "Error in " << kernel_type << " kernel:" << std::endl;
        std::cout << "Maximum flag: " << maximum << " != 2." << std::endl;
    }
}

void check_nodes(const thrust::device_vector<Node>& d_nodes,
                 const thrust::host_vector<Node>& h_reference,
                 const thrust::device_vector<unsigned int>& d_flags,
                 const char kernel_type[])
{
    thrust::host_vector<Node> h_nodes = d_nodes;
    for (int i=0; i<h_reference.size(); i++) {
        if (h_nodes[i].data != h_reference[i].data) {
            std::cout << "Error in " << kernel_type << " kernel:" << std::endl;
            std::cout << "Device node[" << i << "].data != reference."
                      << std::endl;
            std::cout << h_nodes[i].data << " != " << h_reference[i].data
                      << std::endl;
        }
    }
    check_flags(d_flags, kernel_type);
}

void check_data(const thrust::device_vector<float>& d_node_data,
                const thrust::host_vector<Node>& h_reference,
                const thrust::device_vector<unsigned int>& d_flags,
                const char kernel_type[])
{
    thrust::host_vector<float> h_node_data = d_node_data;
    for (int i=0; i<h_reference.size(); i++) {
        if (h_node_data[i] != h_reference[i].data) {
            std::cout << "Error in " << kernel_type << " kernel:" << std::endl;
            std::cout << "Device node[" << i << "].data != reference."
                      << std::endl;
            std::cout << h_node_data[i] << " != " << h_reference[i].data
                      << std::endl;
        }
    }
    check_flags(d_flags, kernel_type);
}

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::setprecision(5);
    std::setfill('0');

    /* Initialize run parameters. */

    unsigned int levels = 17;
    unsigned int N_iter = 10;
    if (argc > 2) {
        levels = (unsigned int) std::strtol(argv[1], NULL, 10);
        N_iter = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    else if (argc > 1) {
        // N needs to be expressable as SUM_{i=0}^{n}(2^i) for our tree
        // generating step, i.e. all bits up to some n set, all others unset.
        levels = (unsigned int) std::strtol(argv[1], NULL, 10);
        if (levels > 32) {
            levels = 32;
        }
    }
    unsigned int N = (1u << levels) - 1; // 2^17 - 1 = 131071.
    std::cout << "Will generate " << N << " nodes in " << levels << " levels."
              << std::endl;
    std::cout << "Will complete " << N_iter << " iterations." << std::endl;


    /* Build tree on host. */

    thrust::host_vector<Node> h_nodes(N);
    thrust::host_vector<NodeNoData> h_nodes_nodata(N);
    thrust::host_vector<float> h_node_data(N);
    thrust::host_vector<float> h_data(1u << (levels-1));

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(h_data.size()),
                      h_data.begin(),
                      random_float_functor());

    // Set up all except final level.
    for (int level=0; level<levels-1; level++) {
        int level_start = (1u << level) - 1;
        int level_end = (1u << (level+1)) - 1;

        for (int i=level_start; i<level_end; i++) {
            int child_index = level_end + 2 * (i - level_start);
            h_nodes[i].left = child_index;
            h_nodes[i].right = child_index + 1;
            h_nodes[child_index].parent = i;
            h_nodes[child_index+1].parent = i;
            h_nodes[i].level = level;

            h_nodes_nodata[i].left = child_index;
            h_nodes_nodata[i].right = child_index + 1;
            h_nodes_nodata[child_index].parent = i;
            h_nodes_nodata[child_index+1].parent = i;
            h_nodes_nodata[i].level = level;
        }
    }

    // Set up final level, including assigning data.
    int final_start = (1u << (levels-1)) - 1;
    for (int i=final_start; i<N; i++) {
        h_nodes[i].left = -1;
        h_nodes[i].right = -1;
        h_nodes[i].data = h_data[i-final_start];
        h_nodes[i].level = levels - 1;

        h_nodes_nodata[i].left = -1;
        h_nodes_nodata[i].right = -1;
        h_node_data[i] = h_data[i-final_start];
        h_nodes_nodata[i].level = levels - 1;
    }

    // Build a reference tree, with correct data, on the CPU.
    // Start at final-but-one level.
    thrust::host_vector<Node> h_nodes_ref = h_nodes;
    for (int level=levels-2; level>=0; level--) {
        int level_start = (1u << level) - 1;
        int level_end = (1u << (level+1)) - 1;

        for (int i=level_start; i<level_end; i++) {
            h_nodes_ref[i].data = min(h_nodes_ref[h_nodes_ref[i].left].data,
                                      h_nodes_ref[h_nodes_ref[i].right].data);
            if (h_nodes_ref[i].data == 0.0f) {
                std::cout << "data at [" << i << "] = " << h_nodes[i].data
                          << ".  Tree construction bug!" << std::endl;
            }
        }
    }

    thrust::device_vector<Node> d_nodes = h_nodes;
    thrust::device_vector<NodeNoData> d_nodes_nodata = h_nodes_nodata;
    thrust::device_vector<float> d_node_data = h_node_data;
    thrust::device_vector<unsigned int> d_flags(N);


    /* Test the kernels. */
    int threads_per_block = 512;
    int blocks = min(112, (N + threads_per_block-1)/threads_per_block);

    for (int i=0; i<N_iter; i++) {
        volatile_node<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, d_flags,
                    "volatile node");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;


        separate_volatile_data<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes_nodata.data()),
            thrust::raw_pointer_cast(d_node_data.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_data(d_node_data, h_nodes_ref, d_flags,
                   "separate volatile data");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_node_data = h_node_data;


        atomic_read<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, d_flags,
                    "atomicAdd()");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;


        atomic_read_conditional<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, d_flags,
                    "conditional atomicAdd()");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;


        asm_read<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, d_flags,
                    "PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;


        asm_read_conditional<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, d_flags,
                    "conditional PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;


        separate_asm_read<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes_nodata.data()),
            thrust::raw_pointer_cast(d_node_data.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_data(d_node_data, h_nodes_ref, d_flags,
                   "separate PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_node_data = h_node_data;


        separate_asm_read_conditional<<<blocks,threads_per_block>>>(
            thrust::raw_pointer_cast(d_nodes_nodata.data()),
            thrust::raw_pointer_cast(d_node_data.data()),
            N,
            final_start,
            thrust::raw_pointer_cast(d_flags.data()));
        check_data(d_node_data, h_nodes_ref, d_flags,
                   "conditional separate PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_node_data = h_node_data;    }
}
