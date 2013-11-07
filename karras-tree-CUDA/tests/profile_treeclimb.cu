#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>

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

int main(int argc, char* argv[]) {
    cudaDeviceProp deviceProp;
    std::ofstream outfile;
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(5);
    outfile.fill('0');

    /* Initialize global parameters. */

    unsigned int max_level = 25;
    unsigned int min_level = 5;
    unsigned int N_iter = 1000;
    unsigned int file_num = 1;
    unsigned int device_id = 0;
    if (argc > 5) {
        min_level = (unsigned int) std::strtol(argv[1], NULL, 10);
        max_level = (unsigned int) std::strtol(argv[2], NULL, 10);
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
        file_num = (unsigned int) std::strtol(argv[4], NULL, 10);
        device_id = (unsigned int) std::strtol(argv[5], NULL, 10);
    }
    else if (argc > 4) {
        min_level = (unsigned int) std::strtol(argv[1], NULL, 10);
        max_level = (unsigned int) std::strtol(argv[2], NULL, 10);
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
        file_num = (unsigned int) std::strtol(argv[4], NULL, 10);
    }
    else if (argc > 3) {
        min_level = (unsigned int) std::strtol(argv[1], NULL, 10);
        max_level = (unsigned int) std::strtol(argv[2], NULL, 10);
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
    }
    else if (argc > 2) {
        // N needs to be expressable as SUM_{i=0}^{n}(2^i) for our tree
        // generating step, i.e. all bits up to some n set, all others unset.
        max_level = (unsigned int) std::strtol(argv[1], NULL, 10);
        N_iter = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    else if (argc > 1) {
        max_level = (unsigned int) std::strtol(argv[1], NULL, 10);
    }

    if (max_level > 32)
        max_level = 32;
    if (min_level < 2)
        min_level = 2;

    std::ostringstream convert;
    std::string file_num_str;
    convert << file_num;
    file_num_str = convert.str();
    std::string file_name = ("profile_treeclimb_" + file_num_str + ".log");

    std::cout << "Will profile with trees of depth " << min_level
              << " to " << max_level << ", making " << N_iter
              << " iterations per tree." << std::endl;
    std::cout << "Running on device " << device_id << std::endl;
    std::cout << "Will save results to " << file_name << std::endl;

    cudaGetDeviceProperties(&deviceProp, device_id);
    cudaSetDevice(device_id);
    // Wipe the file, if it exists.
    outfile.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);
    outfile << "Device " << device_id
                    << ":                 " << deviceProp.name << std::endl;
    outfile << "Starting tree depth:      " << min_level << std::endl;
    outfile << "Finishing tree depth:     " << max_level << std::endl;
    outfile << "Iterations per tree:      " << N_iter << std::endl;
    outfile << "Threads per block:        " << THREADS_PER_BLOCK << std::endl;
    outfile << "Maximum number of blocks: " << NUM_BLOCKS << std::endl;
    outfile << std::endl;
    outfile << std::endl;
    outfile.close();

    for (int levels=min_level; levels<=max_level; levels++) {

        /* Initialize level-specific parameters. */

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float volatile_node_t, separate_volatile_data_t;
        float atomic_read_conditional_t;
        float asm_read_t, asm_read_conditional_t;
        float separate_asm_read_t, separate_asm_read_conditional_t;
        float sm_flags_volatile_node_t, sm_flags_separate_volatile_data_t;
        float elapsed_time;

        std::cout << "Calculating for tree of depth " << levels << "..."
                  << std::endl;

        outfile.open(file_name.c_str(),
                     std::ofstream::out | std::ofstream::app);
        unsigned int N_leaves = 1u << (levels-1);
        outfile << "Will generate " << levels << " levels, with "
                << N_leaves << " leaves and " << N_leaves-1 << " nodes."
                << std::endl;
        outfile << std::endl;
        outfile.close();


        /* Build tree on host. */

        thrust::host_vector<Node> h_nodes(N_leaves-1);
        thrust::host_vector<Leaf> h_leaves(N_leaves);

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
        for (unsigned int height=2; height<(levels-1); height++) {
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
        for (unsigned int height=2; height<(levels-1); height++) {
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


        /* Profile the kernels.
         * NUM_BLOCKS and THREADS_PER_BLOCK are defined in tree_climb_kernels.cuh
         */
        int blocks = min(NUM_BLOCKS,
                         (N_leaves + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);

        for (int i=0; i<N_iter; i++) {
            cudaEventRecord(start);
            volatile_node<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                thrust::raw_pointer_cast(d_leaves.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;
            d_leaves = h_leaves;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            volatile_node_t += elapsed_time;


            cudaEventRecord(start);
            separate_volatile_data<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes_nodata.data()),
                thrust::raw_pointer_cast(d_leaves_nodata.data()),
                thrust::raw_pointer_cast(d_node_data.data()),
                thrust::raw_pointer_cast(d_leaf_data.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            thrust::fill(d_node_data.begin(), d_node_data.end(), 0);
            thrust::fill(d_leaf_data.begin(), d_leaf_data.end(), 0);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            separate_volatile_data_t += elapsed_time;


            cudaEventRecord(start);
            atomic_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                thrust::raw_pointer_cast(d_leaves.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;
            d_leaves = h_leaves;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            atomic_read_conditional_t += elapsed_time;


            cudaEventRecord(start);
            asm_read<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                thrust::raw_pointer_cast(d_leaves.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;
            d_leaves = h_leaves;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            asm_read_t += elapsed_time;


            cudaEventRecord(start);
            asm_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                thrust::raw_pointer_cast(d_leaves.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;
            d_leaves = h_leaves;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            asm_read_conditional_t += elapsed_time;


            cudaEventRecord(start);
            separate_asm_read<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes_nodata.data()),
                thrust::raw_pointer_cast(d_leaves_nodata.data()),
                thrust::raw_pointer_cast(d_node_data.data()),
                thrust::raw_pointer_cast(d_leaf_data.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            thrust::fill(d_node_data.begin(), d_node_data.end(), 0);
            thrust::fill(d_leaf_data.begin(), d_leaf_data.end(), 0);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            separate_asm_read_t += elapsed_time;


            cudaEventRecord(start);
            separate_asm_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes_nodata.data()),
                thrust::raw_pointer_cast(d_leaves_nodata.data()),
                thrust::raw_pointer_cast(d_node_data.data()),
                thrust::raw_pointer_cast(d_leaf_data.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            thrust::fill(d_node_data.begin(), d_node_data.end(), 0);
            thrust::fill(d_leaf_data.begin(), d_leaf_data.end(), 0);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            separate_asm_read_conditional_t += elapsed_time;


            cudaEventRecord(start);
            sm_flags_volatile_node<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                thrust::raw_pointer_cast(d_leaves.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;
            d_leaves = h_leaves;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            sm_flags_volatile_node_t += elapsed_time;

            cudaEventRecord(start);
            sm_flags_separate_volatile_data<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes_nodata.data()),
                thrust::raw_pointer_cast(d_leaves_nodata.data()),
                thrust::raw_pointer_cast(d_node_data.data()),
                thrust::raw_pointer_cast(d_leaf_data.data()),
                N_leaves,
                thrust::raw_pointer_cast(d_data.data()),
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            thrust::fill(d_node_data.begin(), d_node_data.end(), 0);
            thrust::fill(d_leaf_data.begin(), d_leaf_data.end(), 0);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            sm_flags_separate_volatile_data_t += elapsed_time;
        }


        volatile_node_t /= N_iter;
        separate_volatile_data_t /= N_iter;
        atomic_read_conditional_t /= N_iter;
        asm_read_t /= N_iter;
        asm_read_conditional_t /= N_iter;
        separate_asm_read_t /= N_iter;
        separate_asm_read_conditional_t /= N_iter;
        sm_flags_volatile_node_t /= N_iter;
        sm_flags_separate_volatile_data_t /= N_iter;


        /* Write results of this iteration level to file. */

        outfile.open(file_name.c_str(),
                     std::ofstream::out | std::ofstream::app);
        outfile << "Time for volatile node:                              "
                << volatile_node_t << " ms." << std::endl;
        outfile << "Time for separate volatile data:                     "
                << separate_volatile_data_t << " ms." << std::endl;
        outfile << "Time for conditional atomicAdd():                    "
                << atomic_read_conditional_t << " ms." << std::endl;
        outfile << "Time for inline PTX:                                 "
                << asm_read_t << " ms." << std::endl;
        outfile << "Time for conditional inline PTX:                     "
                << asm_read_conditional_t << " ms." << std::endl;
        outfile << "Time for separate data inline PTX:                   "
                << separate_asm_read_t << " ms." << std::endl;
        outfile << "Time for separate data conditional inline PTX:       "
                << separate_asm_read_conditional_t << " ms." << std::endl;
        outfile << "Time for shared memory flags volatile node:          "
                << sm_flags_volatile_node_t << " ms." << std::endl;
        outfile << "Time for shared memory flags separate volatile data: "
                << sm_flags_separate_volatile_data_t << std::endl;
        outfile << std::endl;
        outfile << std::endl;
        outfile.close();
    }
}
