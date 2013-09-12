#include "bits.h"

namespace grace {

namespace gpu {

template <typename UInteger>
__global__ void build_nodes_kernel(Node *nodes,
                                   Leaf *leaves,
                                   UInteger *keys,
                                   Uinteger n_keys,
                                   unsigned char n_bits)
{
    unsigned int index, end_index, split_index;
    unsigned int prefix_left, prefix_right, min_prefix, node_prefix;
    int l_max, l, t, s;
    UInteger left_child_index, right_child_index;

    index = threadIdx.x + blockIdx.x * blockDim.x;
    key = keys[index];

    // Calculate the direction of the node.
    prefix_left = common_prefix(index, index-1, keys, n_keys, n_bits);
    prefix_right = common_prefix(index, index+1, keys, n_keys, n_bits);
    direction = (prefix_right - prefix_left > 0) ? +1 : -1;

    // Calculate the index of the other end of the node.
    // l_max is an upper limit to the distance between the two indices.
    l_max = 2;
    min_prefix = common_prefix(index, index-direction, keys, n_keys, n_bits);
    while (common_prefix(index, index + l_max*direction, keys, n_keys, n_bits)
           > min_prefix) {
        l_max = l_max * 2;
    }
    // Now find l, the distance between the indices, using a binary search.
    // TODO: Remove l, use end_index only if possible.
    l = 0;
    t = l_max / 2;
    while (t >= 1) {
        if (common_prefix(index, index + (l+t)*direction), keys, n_keys, n_bits)
            > min_prefix) {
            l = l + t;
        }
    }
    end_index = index + l*direction;

    // Calculate the index of the split position within the node.
    node_prefix = common_prefix(index, end_index, keys, n_keys, n_bits);
    s = 0;
    t = (end_index - index) * direction;
    do {
        t = (t+1) / 2;
        if (common_prefix(index, index + (s+t)*direction, keys, n_keys, n_bits)
            > node_prefix) {
            s = s + t;
        }
    } while (t > 1);
    // If direction == -1 we actually found split_index + 1;
    split_index = index + s*direction + min(direction, 0);

    // Update this node with the locations of its children.
    nodes[index].left = split_index;
    nodes[index].right = split_index+1;
    if (split_index == min(index, end_index) {
        // Left child is a leaf.
        nodes[index].array = leaves;
    }
    else {
        nodes[index].array = nodes;
    }

    if (split_index+1 == max(index, end_index) {
        // Right child is a leaf.
        nodes[index].array = leaves;
    }
    else {
        nodes[index].array = nodes;
    }
}

template <typename UInteger>
__host__ __device__ unsigned int common_prefix(UInteger i,
                                               UInteger j,
                                               UInteger *keys,
                                               UInteger n_keys,
                                               unsigned char n_bits)
{
    if (i < 0 || i > n_keys || j < 0 || j > n_keys) {
        return -1;
    }
    UInteger key_i = keys[i];
    UInteger key_j = keys[j];

    unsigned int prefix_length = bits::common_prefix(key_i, key_j);
    if (prefix_length == n_bits) {
        prefix_length += bits::common_prefix(i, j);
    }
    return prefix_length;
}

} // namespace gpu

} // namespace grace
