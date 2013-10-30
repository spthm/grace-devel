#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_tree_climb_kernels.cuh"

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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float volatile_node_t, separate_volatile_data_t;
    float atomic_read_t, atomic_read_conditional_t;
    float asm_read_t, asm_read_conditional_t;
    float separate_asm_read_t, separate_asm_read_conditional_t;
    float elapsed_time;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::setprecision(5);
    std::setfill('0');

    /* Initialize run parameters. */

    unsigned int levels = 17;
    unsigned int N_iter = 100;
    if (argc > 1) {
        // N needs to be expressable as SUM_{i=0}^{n}(2^i) for our tree
        // generating step, i.e. all bits up to some n set, all others unset.
        levels = (unsigned int) std::strtol(argv[1], NULL, 10);
        if (levels > 32) {
            levels = 32;
        }
    }
    if (argc > 2) {
        N_iter = (unsigned int) std::strtol(argv[2], NULL, 10);
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


    /* Test the kernels. */

    for (int i=0; i<N_iter; i++) {
        cudaEventRecord(start);
        volatile_node<<<112,512>>>(
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
        separate_volatile_data<<<112,512>>>(
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
        atomic_read<<<112,512>>>(
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
        atomic_read_conditional<<<112,512>>>(
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
        asm_read<<<112,512>>>(
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
        asm_read_conditional<<<112,512>>>(
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
        separate_asm_read<<<112,512>>>(
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
        separate_asm_read_conditional<<<112,512>>>(
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

    std::cout << "Time for volatile Node:                        "
        << volatile_node_t << "ms." << std::endl;
    std::cout << "Time for separate volatile data:               "
        << separate_volatile_data_t << "ms." << std::endl;
    std::cout << "Time for atomicAdd():                          "
        << atomic_read_t << "ms." << std::endl;
    std::cout << "Time for conditional atomicAdd():              "
        << atomic_read_conditional_t << "ms." << std::endl;
    std::cout << "Time for inline PTX:                           "
        << asm_read_t << "ms." << std::endl;
    std::cout << "Time for conditional inline PTX:               "
        << asm_read_conditional_t << "ms." << std::endl;
    std::cout << "Time for separate data inline PTX:             "
        << separate_asm_read_t << "ms." << std::endl;
    std::cout << "Time for separate data conditional inline PTX: "
        << separate_asm_read_conditional_t << "ms." << std::endl;

    // // Starting at final-level-but-one, add data to all nodes.  Ensure no 0s.
    // for (int level=levels-2; level>=0; level--) {
    //     int level_start = (1u << level) - 1;
    //     int level_end = (1u << (level+1)) - 1;

    //     for (int i=level_start; i<level_end; i++) {
    //         h_nodes[i].data = min(h_nodes[h_nodes[i].left].data,
    //                               h_nodes[h_nodes[i].right].data);
    //         if (h_nodes[i].data == 0.0f) {
    //             std::cout << "data at [" << i << "] = " << h_nodes[i].data
    //                       << ".  Tree construction bug!" << std::endl;
    //         }
    //     }
    // }


    // unsigned int minimum = thrust::reduce(d_flags.begin(),
    //                                       d_flags.end(),
    //                                       10u,
    //                                       thrust::minimum<unsigned int>());
    // unsigned int maximum = thrust::reduce(d_flags.begin(),
    //                                       d_flags.end(),
    //                                       0u,
    //                                       thrust::maximum<unsigned int>());
    // std::cout << "Minimum flag: " << minimum << std::endl;
    // std::cout << "Maximum flag: " << maximum << std::endl;
}
