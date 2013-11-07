#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "treeclimb_kernels.cuh"

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
                 const thrust::host_vector<Node>& h_nodes_ref,
                 const char kernel_type[])
{
    thrust::host_vector<Node> h_nodes = d_nodes;
    for (unsigned int i=0; i<h_nodes_ref.size(); i++) {
        if (h_nodes[i].data != h_nodes_ref[i].data) {
            std::cout << "Error in " << kernel_type << " kernel:" << std::endl;
            std::cout << "Device node[" << i << "].data != reference."
                      << std::endl;
            std::cout << h_nodes[i].data << " != " << h_nodes_ref[i].data
                      << std::endl;
        }
    }
}

void check_data(const thrust::device_vector<float>& d_node_data,
                const thrust::host_vector<Node>& h_nodes_ref,
                const char kernel_type[])
{
    thrust::host_vector<float> h_node_data = d_node_data;
    for (unsigned int i=0; i<h_nodes_ref.size(); i++) {
        if (h_node_data[i] != h_nodes_ref[i].data) {
            std::cout << "Error in " << kernel_type << " kernel:" << std::endl;
            std::cout << "Device node[" << i << "].data != reference."
                      << std::endl;
            std::cout << h_node_data[i] << " != " << h_nodes_ref[i].data
                      << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::setprecision(5);
    std::setfill('0');

    /* Initialize run parameters. */

    unsigned int depth = 16;
    unsigned int N_iter = 50;
    if (argc > 2) {
        depth = (unsigned int) std::strtol(argv[1], NULL, 10);
        N_iter = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    else if (argc > 1) {
        // N needs to be expressable as SUM_{i=0}^{n}(2^i) for our tree
        // generating step, i.e. all bits up to some n set, all others unset.
        depth = (unsigned int) std::strtol(argv[1], NULL, 10);
        if (depth > 31) {
            depth = 31;
        }
    }
    unsigned int N_leaves = (1u << depth);
    std::cout << "Will generate a tree with 2^" << depth << " = " << N_leaves
              << " leaves and 2^" << depth << " -1 nodes." << std::endl;
    std::cout << "Total number of nodes + leaves = " << 2*N_leaves - 1
              << std::endl;
    std::cout << "Will complete " << N_iter << " iterations." << std::endl;


    /* Build trees on host. */

    thrust::host_vector<Node> h_nodes(N_leaves-1);
    thrust::host_vector<Leaf> h_leaves(N_leaves);

    // Fill data vector with random floats in [0,1).
    thrust::host_vector<float> h_data(N_leaves);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(h_data.size()),
                      h_data.begin(),
                      random_float_functor());

    // Set up bottom level (where all nodes connect to leaves).
    for (unsigned int left=1; left<N_leaves-1; left+=4) {
        unsigned int right = left + 1;

        h_nodes[left].left = left - 1;
        h_nodes[left].right = left;
        h_nodes[left].far_end = left - 1;
        h_nodes[left].left_is_leaf = true;
        h_nodes[left].right_is_leaf = true;

        h_nodes[right].left = right;
        h_nodes[right].right = right + 1;
        h_nodes[right].far_end = right + 1;
        h_nodes[right].left_is_leaf = true;
        h_nodes[right].right_is_leaf = true;

        h_leaves[left-1].parent = h_leaves[left].parent = left;
        h_leaves[right].parent = h_leaves[right+1].parent = right;
    }
    // Set up all except bottom and top levels, starting at bottom-but-one.
    for (unsigned int height=2; height<depth; height++) {
        for (unsigned int left=(1u<<height)-1;
                          left<N_leaves-1;
                          left+=1u<<(height+1)) {
            unsigned int right = left + 1;
            unsigned int left_split = (2*left - (1u<<height)) / 2;
            unsigned int right_split = left_split + (1u<<height);

            h_nodes[left].left = left_split;
            h_nodes[left].right = left_split + 1;
            h_nodes[left].far_end = left - (1u<<height) + 1;
            h_nodes[left].left_is_leaf = false;
            h_nodes[left].right_is_leaf = false;

            h_nodes[right].left = right_split;
            h_nodes[right].right = right_split + 1;
            h_nodes[right].far_end = right + (1u<<height) - 1;
            h_nodes[right].left_is_leaf = false;
            h_nodes[right].right_is_leaf = false;

            h_nodes[left_split].parent = h_nodes[left_split+1].parent = left;
            h_nodes[right_split].parent = h_nodes[right_split+1].parent = right;
        }
    }
    // Set up root node and link children to it.
    h_nodes[0].left = N_leaves/2 - 1;
    h_nodes[0].right = N_leaves/2;
    h_nodes[0].far_end = N_leaves - 1;
    h_nodes[0].left_is_leaf = false;
    h_nodes[0].right_is_leaf = false;
    h_nodes[N_leaves/2 - 1].parent = h_nodes[N_leaves/2].parent = 0;


    // Build the leaves and nodes which contain no data.
    thrust::host_vector<NodeNoData> h_nodes_nodata(N_leaves-1);
    thrust::host_vector<LeafNoData> h_leaves_nodata(N_leaves);
    for (unsigned int i=0; i<N_leaves-1; i++) {
        h_leaves_nodata[i].parent = h_leaves[i].parent;

        h_nodes_nodata[i].left = h_nodes[i].left;
        h_nodes_nodata[i].right = h_nodes[i].right;
        h_nodes_nodata[i].parent = h_nodes[i].parent;
        h_nodes_nodata[i].far_end = h_nodes[i].far_end;
        h_nodes_nodata[i].left_is_leaf = h_nodes[i].left_is_leaf;
        h_nodes_nodata[i].right_is_leaf = h_nodes[i].right_is_leaf;
    }
    h_leaves_nodata[N_leaves-1].parent = h_leaves[N_leaves-1].parent;


    // Build a reference tree with correct data.
    // Start with bottom level, which connect directly to leaves.
    thrust::host_vector<Node> h_nodes_ref = h_nodes;
    for (unsigned int left=1; left<N_leaves-1; left+=4) {
        unsigned int right = left + 1;
        h_nodes_ref[left].data = min(h_data[left-1],
                                     h_data[left]);
        h_nodes_ref[right].data = min(h_data[right],
                                      h_data[right+1]);
    }
    // Assign data to all other levels except the root.
    for (unsigned int height=2; height<depth; height++) {
        for (unsigned int left=(1u<<height)-1;
                          left<N_leaves-1;
                          left+=1u<<(height+1)) {
            unsigned int right = left + 1;
            h_nodes_ref[left].data =
                min(h_nodes_ref[h_nodes_ref[left].left].data,
                    h_nodes_ref[h_nodes_ref[left].right].data);
            h_nodes_ref[right].data =
                min(h_nodes_ref[h_nodes_ref[right].left].data,
                    h_nodes_ref[h_nodes_ref[right].right].data);
        }
    }
    // Assign data to the root node.
    h_nodes_ref[0].data = min(h_nodes_ref[h_nodes_ref[0].left].data,
                              h_nodes_ref[h_nodes_ref[0].right].data);


    thrust::device_vector<Node> d_nodes = h_nodes;
    thrust::device_vector<Leaf> d_leaves = h_leaves;
    thrust::device_vector<NodeNoData> d_nodes_nodata = h_nodes_nodata;
    thrust::device_vector<LeafNoData> d_leaves_nodata = h_leaves_nodata;
    thrust::device_vector<float> d_data = h_data;
    thrust::device_vector<float> d_node_data(N_leaves-1);
    thrust::device_vector<float> d_leaf_data(N_leaves);
    thrust::device_vector<unsigned int> d_flags(N_leaves-1);


    /* Test the kernels.
     * THREADS_PER_BLOCK and NUM_BLOCKS are defined in tree_climb_kernels.cuh
     */
    int blocks = min(NUM_BLOCKS,
                     (N_leaves + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);

    for (int i=0; i<N_iter; i++) {
        volatile_node<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_leaves.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, "volatile");
        check_flags(d_flags, "volatile");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;
        d_leaves = h_leaves;


        separate_volatile_data<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes_nodata.data()),
            thrust::raw_pointer_cast(d_leaves_nodata.data()),
            thrust::raw_pointer_cast(d_node_data.data()),
            thrust::raw_pointer_cast(d_leaf_data.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_data(d_node_data, h_nodes_ref, "separate volatile");
        check_flags(d_flags, "separate volatile");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        thrust::fill(d_node_data.begin(), d_node_data.end(), 0);
        thrust::fill(d_leaf_data.begin(), d_leaf_data.end(), 0);


        atomic_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_leaves.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, "conditional atomicAdd()");
        check_flags(d_flags, "conditional atomicAdd()");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;
        d_leaves = h_leaves;


        asm_read<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_leaves.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, "PTX load");
        check_flags(d_flags, "PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;
        d_leaves = h_leaves;


        asm_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_leaves.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, "conditional PTX load");
        check_flags(d_flags, "conditional PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;
        d_leaves = h_leaves;


        separate_asm_read<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes_nodata.data()),
            thrust::raw_pointer_cast(d_leaves_nodata.data()),
            thrust::raw_pointer_cast(d_node_data.data()),
            thrust::raw_pointer_cast(d_leaf_data.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_data(d_node_data, h_nodes_ref, "separate PTX load");
        check_flags(d_flags, "separate PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        thrust::fill(d_node_data.begin(), d_node_data.end(), 0);
        thrust::fill(d_leaf_data.begin(), d_leaf_data.end(), 0);


        separate_asm_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes_nodata.data()),
            thrust::raw_pointer_cast(d_leaves_nodata.data()),
            thrust::raw_pointer_cast(d_node_data.data()),
            thrust::raw_pointer_cast(d_leaf_data.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_data(d_node_data, h_nodes_ref, "conditional separate PTX load");
        check_flags(d_flags, "vconditional separate PTX load");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        thrust::fill(d_node_data.begin(), d_node_data.end(), 0);
        thrust::fill(d_leaf_data.begin(), d_leaf_data.end(), 0);


        sm_flags_volatile_node<<<blocks,THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_leaves.data()),
            N_leaves,
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_flags.data()));
        check_nodes(d_nodes, h_nodes_ref, "sm volatile");
        thrust::fill(d_flags.begin(), d_flags.end(), 0);
        d_nodes = h_nodes;
        d_leaves = h_leaves;
    }
}
