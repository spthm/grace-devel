#define THREADS_PER_BLOCK 512
// M2090: 1536 threads/MP
//        8 blocks/MP
//        16 MPs
//        512 threads/block -> maximum of 3 *resident* blocks/SM
//                          -> 16 * 3 = 48 blocks
// GTX 670: 2048 threads/MP
//          16 blocks/MP
//          7 MPs
//          512 threads/block -> maximum of 4 *resident* blocks/SM
//                            -> 7 * 4 = 28 blocks
#if __CUDA_ARCH__ >= 300
#define NUM_BLOCKS 28
#else
#define NUM_BLOCKS 48
#endif

struct Node
{
    int left;
    int right;
    int parent;

    bool left_is_leaf;
    bool right_is_leaf;

    unsigned int level;

    float data;
};

struct Leaf
{
    int parent;

    float data;
};

struct NodeNoData
{
    int left;
    int right;
    int parent;

    bool left_is_leaf;
    bool right_is_leaf;

    unsigned int level;
};

struct LeafNoData
{
    int parent;
};


__global__ void volatile_node(volatile Node* nodes,
                              volatile Leaf* leaves,
                              const unsigned int n_leaves,
                              const float* raw_data,
                              unsigned int* flags)
{
    int tid, index, left, right;
    float data;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_leaves)
    {
        leaves[tid].data = raw_data[tid];

        __threadfence();

        // Move up the tree.
        index = leaves[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            if (nodes[index].left_is_leaf) {
                data = min(leaves[left].data, leaves[right].data);
            }
            else {
                data = min(nodes[left].data, nodes[right].data);
            }

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

__global__ void separate_volatile_data(const NodeNoData* nodes,
                                       const LeafNoData* leaves,
                                       volatile float* node_data,
                                       volatile float* leaf_data,
                                       const unsigned int n_leaves,
                                       const float* raw_data,
                                       unsigned int* flags)
{
    int tid, index, left, right;
    float data;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_leaves)
    {
        leaf_data[tid] = raw_data[tid];

        __threadfence();

        // Move up the tree.
        index = leaves[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            if (nodes[index].left_is_leaf) {
                data = min(leaf_data[left], leaf_data[right]);
            }
            else {
                data = min(node_data[left], node_data[right]);
            }

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

__global__ void atomic_read_conditional(Node* nodes,
                                        Leaf* leaves,
                                        const unsigned int n_leaves,
                                        const float* raw_data,
                                        unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_leaves)
    {
        leaves[tid].data = raw_data[tid];

        __threadfence();

        // Move up the tree.
        index = leaves[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            if (nodes[index].left_is_leaf) {
                data_left = leaves[left].data;
                data_right = leaves[right].data;
                // Check if the data is exactly 0.  Re-read from global/L2 if so.
                // If it is still zero, then the actual data must be exactly 0.
                if (data_left == 0.0f)
                    data_left = atomicAdd(&leaves[left].data, 0.0f);
                if (data_right == 0.0f)
                    data_right = atomicAdd(&leaves[right].data, 0.0f);
            }
            else {
                data_left = nodes[left].data;
                data_right = nodes[right].data;
                if (data_left == 0.0f)
                    data_left = atomicAdd(&nodes[left].data, 0.0f);
                if (data_right == 0.0f)
                    data_right = atomicAdd(&nodes[right].data, 0.0f);
            }

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

__global__ void asm_read(Node* nodes,
                         Leaf* leaves,
                         const unsigned int n_leaves,
                         const float* raw_data,
                         unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_leaves)
    {
        leaves[tid].data = raw_data[tid];

        __threadfence();

        // Move up the tree.
        index = leaves[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            if (nodes[index].left_is_leaf) {
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&leaves[left].data));
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&leaves[right].data));
            }
            else {
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&nodes[left].data));
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&nodes[right].data));
            }

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

__global__ void asm_read_conditional(Node* nodes,
                                     Leaf* leaves,
                                     const unsigned int n_leaves,
                                     const float* raw_data,
                                     unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_leaves)
    {
        leaves[tid].data = raw_data[tid];

        __threadfence();

        // Move up the tree.
        index = leaves[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            if (nodes[index].left_is_leaf) {
                data_left = leaves[left].data;
                data_right = leaves[right].data;

                if (data_left == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&leaves[left].data));
                if (data_right == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&leaves[right].data));
            }
            else {
                data_left = nodes[left].data;
                data_right = nodes[right].data;

                if (data_left == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&nodes[left].data));
                if (data_right == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&nodes[right].data));
            }

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

__global__ void separate_asm_read(const NodeNoData* nodes,
                                  const LeafNoData* leaves,
                                  float* node_data,
                                  float* leaf_data,
                                  const unsigned int n_leaves,
                                  const float* raw_data,
                                  unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_leaves)
    {
        leaf_data[tid] = raw_data[tid];

        __threadfence();

        // Move up the tree.
        index = leaves[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            if (nodes[index].left_is_leaf) {
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&leaf_data[left]));
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&leaf_data[right]));
            }
            else {
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&node_data[left]));
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&node_data[right]));
            }

            data = min(data_left, data_right);

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

__global__ void separate_asm_read_conditional(const NodeNoData* nodes,
                                              const LeafNoData* leaves,
                                              float* node_data,
                                              float* leaf_data,
                                              const unsigned int n_leaves,
                                              const float* raw_data,
                                              unsigned int* flags)
{
    int tid, index, left, right;
    float data, data_left, data_right;
    bool first_arrival;

    tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < n_leaves)
    {
        leaf_data[tid] = raw_data[tid];

        __threadfence();

        // Move up the tree.
        index = leaves[tid].parent;
        first_arrival = (atomicAdd(&flags[index], 1) == 0);

        // If we are the second thread to reach this node then process it.
        while (!first_arrival)
        {
            left = nodes[index].left;
            right = nodes[index].right;

            if (nodes[index].left_is_leaf) {
                data_left = leaf_data[left];
                data_right = leaf_data[right];

                if (data_left == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&leaf_data[left]));
                if (data_right == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&leaf_data[right]));
            }
            else {
                data_left = node_data[left];
                data_right = node_data[right];

                if (data_left == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_left) : "l"(&node_data[left]));
                if (data_right == 0.0f)
                    asm("ld.global.cg.f32 %0, [%1];" : "=f"(data_right) : "l"(&node_data[right]));
            }

            data = min(data_left, data_right);

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

// __global__ void sm_flags_volatile_node(volatile Node* nodes,
//                                        volatile Leaf* leaves,
//                                        const unsigned int n_leaves,
//                                        const float* raw_data,
//                                        unsigned int *g_flags)
// {
//     int tid, index, left, right;
//     float data;
//     bool first_arrival;

//     __shared__ unsigned int sm_flags[THREADS_PER_BLOCK];

//     tid = threadIdx.x + blockIdx.x * blockDim.x;
// }
