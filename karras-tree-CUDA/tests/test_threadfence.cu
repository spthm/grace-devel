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
    int level;

    volatile float data;
};

// Need Node(.data) to be marked as volatile for this to work correctly.

// This is to prevent L1 cache reads of old data even after another thread
// has flushed new data to global memory.

//** Could improve performance by having 'data' its own array (i.e. the
//** read/writes to the other variables within Node can be still optimized).

//* Using atomics forces a global read(?), but would then require
//* a second write.  E.g. atomicAdd(&node[i].data, 0.0f).
//* Need to profile that on Fermi AND Kepler.

// Alternatively, a hacky "keep polling until != 0.0f" could be done, which
// most of the time will evaluate to false on the first attempt.

//*** OR, read, and if == 0.0f, then atomicAdd as above.
//*** if (data == 0.0f) should be quicker than always doing atomicAdd.

//* Inline PTX on the data = node.data to specify skipping of L1?  Or do this
//* in the case that node.data == 0.0f.  That may be the fastest solution!

// Declaring the variable in the struct as volatile seems like a bad idea
// for all use cases *except* construction (e.g. in tracing it may produce
// a large performance hit).

// Ask on stackoverflow.  If __threadfence() guarantees that writes are flushed
// up the memory hierarchy, but there can be no guarantee that a thread will
// *read* the new value *after* this, then what is the point?

// See:
// http://stackoverflow.com/questions/14484739/
// http://stackoverflow.com/questions/5540217
// https://devtalk.nvidia.com/default/topic/490973/ (second page)
// https://devtalk.nvidia.com/default/topic/489987/
// http://stackoverflow.com/questions/11275744

__global__ void propagate_data(Node *nodes,
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
