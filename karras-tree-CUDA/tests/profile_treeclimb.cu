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
    if (argc > 3) {
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

    std::cout << "Will profile with trees of depth " << min_level
              << " to " << max_level << ", making " << N_iter
              << " iterations per tree." << std::endl;

    cudaGetDeviceProperties(&deviceProp, 0);
    // Wipe the file, if it exists.
    outfile.open("profile_treeclimb_results.log",
                 std::ofstream::out | std::ofstream::trunc);
    outfile << "Device 0:             " << deviceProp.name << std::endl;
    outfile << "Starting tree depth:  " << min_level << std::endl;
    outfile << "Finishing tree depth: " << max_level << std::endl;
    outfile << "Iterations per tree:  " << N_iter << std::endl;
    outfile << std::endl;
    outfile << std::endl;
    outfile.close();

    for (int levels=min_level; levels<=max_level; levels++) {

        /* Initialize level-specific parameters. */

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float volatile_node_t, separate_volatile_data_t;
        float atomic_read_t, atomic_read_conditional_t;
        float asm_read_t, asm_read_conditional_t;
        float separate_asm_read_t, separate_asm_read_conditional_t;
        float elapsed_time;

        std::cout << "Calculating for tree of depth " << levels << "..."
                  << std::endl;

        outfile.open("profile_treeclimb_results.log",
                     std::ofstream::out | std::ofstream::app);
        unsigned int N = (1u << levels) - 1;
        outfile << "Will generate " << N << " nodes in "
                  << levels << " levels." << std::endl;
        outfile << std::endl;
        outfile.close();


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
                h_nodes[i].left = h_nodes_nodata[i].left = child_index;
                h_nodes[i].right = h_nodes_nodata[i].right = child_index + 1;
                h_nodes[child_index].parent = h_nodes[child_index+1].parent = i;
                h_nodes_nodata[child_index].parent = i;
                h_nodes_nodata[child_index+1].parent = i;
                h_nodes[i].level = h_nodes_nodata[i].level = level;
            }
        }

        // Set up final level, including assigning data.
        int final_start = (1u << (levels-1)) - 1;
        for (int i=final_start; i<N; i++) {
            h_nodes[i].left = h_nodes[i].right = -1;
            h_nodes_nodata[i].left = h_nodes_nodata[i].right = -1;
            h_nodes[i].data = h_node_data[i] = h_data[i-final_start];
            h_nodes[i].level = h_nodes_nodata[i].level = levels - 1;
        }

        thrust::device_vector<Node> d_nodes = h_nodes;
        thrust::device_vector<NodeNoData> d_nodes_nodata = h_nodes_nodata;
        thrust::device_vector<float> d_node_data = h_node_data;
        thrust::device_vector<unsigned int> d_flags(N);


        /* Test the kernels.
         * NUM_BLOCKS and THREADS_PER_BLOCK are defined in tree_climb_kernels.cuh
         */
        int blocks = min(NUM_BLOCKS,
                         (N + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);

        for (int i=0; i<N_iter; i++) {
            cudaEventRecord(start);
            volatile_node<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            volatile_node_t += elapsed_time;


            cudaEventRecord(start);
            separate_volatile_data<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes_nodata.data()),
                thrust::raw_pointer_cast(d_node_data.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_node_data = h_node_data;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            separate_volatile_data_t += elapsed_time;


            cudaEventRecord(start);
            atomic_read<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            atomic_read_t += elapsed_time;


            cudaEventRecord(start);
            atomic_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            atomic_read_conditional_t += elapsed_time;


            cudaEventRecord(start);
            asm_read<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            asm_read_t += elapsed_time;


            cudaEventRecord(start);
            asm_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_nodes = h_nodes;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            asm_read_conditional_t += elapsed_time;


            cudaEventRecord(start);
            separate_asm_read<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes_nodata.data()),
                thrust::raw_pointer_cast(d_node_data.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_node_data = h_node_data;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            separate_asm_read_t += elapsed_time;


            cudaEventRecord(start);
            separate_asm_read_conditional<<<blocks,THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_nodes_nodata.data()),
                thrust::raw_pointer_cast(d_node_data.data()),
                N,
                final_start,
                thrust::raw_pointer_cast(d_flags.data()));
            cudaEventRecord(stop);

            thrust::fill(d_flags.begin(), d_flags.end(), 0);
            d_node_data = h_node_data;

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            separate_asm_read_conditional_t += elapsed_time;
        }


        volatile_node_t /= N_iter;
        separate_volatile_data_t /= N_iter;
        atomic_read_t /= N_iter;
        atomic_read_conditional_t /= N_iter;
        asm_read_t /= N_iter;
        asm_read_conditional_t /= N_iter;
        separate_asm_read_t /= N_iter;
        separate_asm_read_conditional_t /= N_iter;


        /* Write results of this iteration level to file. */

        outfile.open("profile_treeclimb_results.log",
                     std::ofstream::out | std::ofstream::app);
        outfile << "Time for volatile Node:                        "
            << volatile_node_t << " ms." << std::endl;
        outfile << "Time for separate volatile data:               "
            << separate_volatile_data_t << " ms." << std::endl;
        outfile << "Time for atomicAdd():                          "
            << atomic_read_t << " ms." << std::endl;
        outfile << "Time for conditional atomicAdd():              "
            << atomic_read_conditional_t << " ms." << std::endl;
        outfile << "Time for inline PTX:                           "
            << asm_read_t << " ms." << std::endl;
        outfile << "Time for conditional inline PTX:               "
            << asm_read_conditional_t << " ms." << std::endl;
        outfile << "Time for separate data inline PTX:             "
            << separate_asm_read_t << " ms." << std::endl;
        outfile << "Time for separate data conditional inline PTX: "
            << separate_asm_read_conditional_t << " ms." << std::endl;
        outfile << std::endl;
        outfile << std::endl;
        outfile.close();
    }
}
