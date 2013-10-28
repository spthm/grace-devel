#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Node
{
    int left;
    int right;
    int parent;

    float data;
};

__global__ void volatile_node(volatile Node *nodes,
                              const unsigned int n_nodes,
                              const unsigned int start,
                              unsigned int* flags)
{
    int tid, index, left, right;
    float data;
    bool first_arrival;

    tid = start + threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_nodes)
    {
        // We start at a node with a full data section; modify its flag
        // accordingly.
        flags[tid] = 2;

        // Immediately move up the tree.
        index = nodes[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            data = min(nodes[left].data, nodes[right].data);

            nodes[index].data = data;

            if (index == 0) {
                // Root node processed, so all nodes processed.
                return;
            }

            // Ensure above global write is visible to all device threads
            // before setting flag for the parent.
            __threadfence();

            index = nodes[index].parent;
            first_arrival = (atomicAdd(&flags[index], 1) == 0);
        }
        tid += blockDim.x*gridDim.x;
    }
    return;
}

__global__ void separate_volatile_data(const Node *nodes,
                                       volatile float* node_data,
                                       const unsigned int n_nodes,
                                       const unsigned int start,
                                       unsigned int* flags)
{
    int tid, index, left, right;
    float data;
    bool first_arrival;

    tid = start + threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_nodes)
    {
        // We start at a node with a full data section; modify its flag
        // accordingly.
        flags[tid] = 2;

        // Immediately move up the tree.
        index = nodes[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            data = min(node_data[left], node_data[right]);

            node_data[index] = data;

            if (index == 0) {
                // Root node processed, so all nodes processed.
                return;
            }

            // Ensure above global write is visible to all device threads
            // before setting flag for the parent.
            __threadfence();

            index = nodes[index].parent;
            first_arrival = (atomicAdd(&flags[index], 1) == 0);
        }
        tid += blockDim.x*gridDim.x;
    }
    return;
}

__global__ void atomic_read(Node *nodes,
                            const unsigned int n_nodes,
                            const unsigned int start,
                            unsigned int* flags)
{
    int tid, index, left, right;
    float data;
    bool first_arrival;

    tid = start + threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_nodes)
    {
        // We start at a node with a full data section; modify its flag
        // accordingly.
        flags[tid] = 2;

        // Immediately move up the tree.
        index = nodes[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            data = min(atomicAdd(&nodes[left].data, 0.0f),
                       atomicAdd(&nodes[right].data, 0.0f));

            nodes[index].data = data;

            if (index == 0) {
                // Root node processed, so all nodes processed.
                return;
            }

            // Ensure above global write is visible to all device threads
            // before setting flag for the parent.
            __threadfence();

            index = nodes[index].parent;
            first_arrival = (atomicAdd(&flags[index], 1) == 0);
        }
        tid += blockDim.x*gridDim.x;
    }
    return;
}

__global__ void atomic_read_conditional(Node *nodes,
                                        const unsigned int n_nodes,
                                        const unsigned int start,
                                        unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = start + threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_nodes)
    {
        // We start at a node with a full data section; modify its flag
        // accordingly.
        flags[tid] = 2;

        // Immediately move up the tree.
        index = nodes[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            data_left = nodes[left].data;
            data_right = nodes[right].data;
            // Check if the data is exactly 0.  Re-read from global/L2 if so.
            // If it is still zero, then the actual data must be exactly 0.
            if (data_left == 0.0f)
                data_left = atomicAdd(&nodes[left].data, 0.0f);
            if (data_right == 0.0f)
                data_right = atomicAdd(&nodes[right].data, 0.0f);

            data = min(data_left, data_right);

            nodes[index].data = data;

            if (index == 0) {
                // Root node processed, so all nodes processed.
                return;
            }

            // Ensure above global write is visible to all device threads
            // before setting flag for the parent.
            __threadfence();

            index = nodes[index].parent;
            first_arrival = (atomicAdd(&flags[index], 1) == 0);
        }
        tid += blockDim.x*gridDim.x;
    }
    return;
}

__global__ void asm_read(Node *nodes,
                         const unsigned int n_nodes,
                         const unsigned int start,
                         unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = start + threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_nodes)
    {
        // We start at a node with a full data section; modify its flag
        // accordingly.
        flags[tid] = 2;

        // Immediately move up the tree.
        index = nodes[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "r"(&nodes[left].data));
            asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "r"(&nodes[right].data));

            data = min(data_left, data_right);

            nodes[index].data = data;

            if (index == 0) {
                // Root node processed, so all nodes processed.
                return;
            }

            // Ensure above global write is visible to all device threads
            // before setting flag for the parent.
            __threadfence();

            index = nodes[index].parent;
            first_arrival = (atomicAdd(&flags[index], 1) == 0);
        }
        tid += blockDim.x*gridDim.x;
    }
    return;
}

__global__ void asm_read_conditional(Node *nodes,
                                     const unsigned int n_nodes,
                                     const unsigned int start,
                                     unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = start + threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_nodes)
    {
        // We start at a node with a full data section; modify its flag
        // accordingly.
        flags[tid] = 2;

        // Immediately move up the tree.
        index = nodes[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            data_left = nodes[left].data;
            data_right = nodes[right].data;

            if (data_left == 0.0f)
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "r"(&nodes[left].data));
            if (data_right == 0.0f)
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "r"(&nodes[right].data));

            data = min(data_left, data_right);

            nodes[index].data = data;

            if (index == 0) {
                // Root node processed, so all nodes processed.
                return;
            }

            // Ensure above global write is visible to all device threads
            // before setting flag for the parent.
            __threadfence();

            index = nodes[index].parent;
            first_arrival = (atomicAdd(&flags[index], 1) == 0);
        }
        tid += blockDim.x*gridDim.x;
    }
    return;
}

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
    std::ofstream outfile;
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(9);

    unsigned int levels = 17;
    unsigned int N = (1u << levels) - 1; // 2^17 - 1 = 131071.
    if (argc > 1) {
        // N needs to be expressable as SUM_{i=0}^{n}(2^i) for our tree
        // generating step, i.e. all bits up to some n set, all others unset.
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
        if (__builtin_clz(N) == 0) {
            levels = 32;
            N = ~0u;
        }
        else {
            // Round up.
            levels = sizeof(N) * CHAR_BIT - __builtin_clz(N);
            N = (1u << levels) - 1;
        }
    }
    std::cout << "Will generate " << N << " nodes, " << levels << " levels."
              << std::endl;

    thrust::host_vector<Node> h_nodes(N);
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
            h_nodes[child_index].parent = h_nodes[child_index+1].parent = i;
            h_nodes[i].level = level;
        }
    }

    // Set up final level, including assigning data and setting flags.
    int final_start = (1u << (levels-1)) - 1;
    for (int i=final_start; i<N; i++) {
        h_nodes[i].left = h_nodes[i].right = -1;
        h_nodes[i].data = h_data[i-final_start];
        h_nodes[i].level = levels - 1;
    }

    thrust::device_vector<Node> d_nodes = h_nodes;
    thrust::device_vector<unsigned int> d_flags(N);
    propagate_data<<<112,512>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        N,
        final_start,
        thrust::raw_pointer_cast(d_flags.data()));
    h_nodes = d_nodes;

    outfile.open("threadfence.txt");
    for (int i=0; i<N; i++) {
        outfile << "i:      " << i << std::endl;
        outfile << "level:  " << h_nodes[i].level << std::endl;
        outfile << "left:   " << h_nodes[i].left << std::endl;
        outfile << "right:  " << h_nodes[i].right << std::endl;
        outfile << "parent: " << h_nodes[i].parent << std::endl;
        outfile << "data:   " << h_nodes[i].data << std::endl;
    }
    outfile.close();

    // Starting at final-level-but-one, add data to all nodes.  Ensure no 0s.
    for (int level=levels-2; level>=0; level--) {
        int level_start = (1u << level) - 1;
        int level_end = (1u << (level+1)) - 1;

        for (int i=level_start; i<level_end; i++) {
            h_nodes[i].data = min(h_nodes[h_nodes[i].left].data,
                                  h_nodes[h_nodes[i].right].data);
            if (h_nodes[i].data == 0.0f) {
                std::cout << "data at [" << i << "] = " << h_nodes[i].data
                          << ".  Tree construction bug!" << std::endl;
            }
        }
    }


    unsigned int minimum = thrust::reduce(d_flags.begin(),
                                          d_flags.end(),
                                          10u,
                                          thrust::minimum<unsigned int>());
    unsigned int maximum = thrust::reduce(d_flags.begin(),
                                          d_flags.end(),
                                          0u,
                                          thrust::maximum<unsigned int>());
    std::cout << "Minimum flag: " << minimum << std::endl;
    std::cout << "Maximum flag: " << maximum << std::endl;
}
