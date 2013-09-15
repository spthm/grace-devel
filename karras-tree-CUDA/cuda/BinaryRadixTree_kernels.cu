#include "Nodes.h"
#include "bits.h"
#include "morton.h"

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

    // Index of this node.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n_keys) {
        key = keys[index];

        // direction == +1 => index is the first key in the node.
        // direction == -1 => index is the last key in the node.
        prefix_left = common_prefix(index, index-1, keys, n_keys, n_bits);
        prefix_right = common_prefix(index, index+1, keys, n_keys, n_bits);
        direction = (prefix_right - prefix_left > 0) ? +1 : -1;

        /* Calculate the index of the other end of the node. */
        // l_max is an upper limit to the distance between the two indices.
        l_max = 2;
        min_prefix = common_prefix(index, index-direction, keys, n_keys, n_bits);
        while (common_prefix(index, index + l_max*direction, keys, n_keys, n_bits)
               > min_prefix) {
            l_max = l_max * 2;
        }
        // Now find l, the distance between the indices, using a binary search.
        // (i.e. one bit at a time.)
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

        /* Calculate the index of the split position within the node. */
        node_prefix = common_prefix(index, end_index, keys, n_keys, n_bits);
        s = 0;
        t = (end_index - index) * direction;
        do {
            // t = ceil(t/2.)
            t = (t+1) / 2;
            if (common_prefix(index, index + (s+t)*direction, keys, n_keys, n_bits)
                > node_prefix) {
                s = s + t;
            }
        } while (t > 1);
        // If direction == -1 we actually found split_index + 1;
        split_index = index + s*direction + min(direction, 0);

        /* Update this node with the locations of its children. */
        nodes[index].left = split_index;
        nodes[index].right = split_index+1;
        if (split_index == min(index, end_index) {
            // Left child is a leaf.
            nodes[index].left_leaf_flag = true;
            leaves[split_index].parent = index;
        }
        else {
            nodes[index].left_leaf_flag = false;
            nodes[split_index].parent = index;
        }

        if (split_index+1 == max(index, end_index) {
            // Right child is a leaf.
            nodes[index].right_leaf_flag = true;
            leaves[split_index+1].parent = index;
        }
        else {
            nodes[index].right_leaf_flag = false;
            nodes[split_index+1].parent = index;
        }
    }
}

template <typename UInteger>
__global__ void find_AABBs_kernel(Node *nodes
                                  Leaf *leaves
                                  UInteger n_leaves,
                                  float *positions,
                                  float *extent,
                                  unsigned char *AABB_flags)
{
    unsigned int index, left_index, right_index;
    float x_min, y_min, z_min
    float x_max, y_max, z_max;
    float r;
    float *left_array, *right_array;

    // Leaf index.
    index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n_leaves) {
        // Find the AABB of each leaf (i.e. each primitive) and write it.
        r = extent[index];
        x_min = positions[index*3 + 0] - r;
        y_min = positions[index*3 + 1] - r;
        z_min = positions[index*3 + 2] - r;

        x_max = x_min + 2*r;
        y_max = y_max + 2*r;
        z_max = z_max + 2*r;

        leaves[index].bottom = {x_min, y_min, z_min};
        leaves[index].top = {x_max, y_max, z_max}:

        // Travel up the tree.  The second thread to reach a node writes
        // its AABB based on those of its children.  The first exists the loop.
        index = leaves[index].parent;
        while true {
            if (atomicAdd(&AABB_flags[index], 1) == 0) {
                break;
            }
            else {
                left_array = (nodes[index].left_leaf_flag) ? leaf : nodes;
                right_array = (nodes[index].right_leaf_flag)? leaf: nodes;

                left_index = nodes[index].left;
                right_index = nodes[index].right;

                x_min = min(left_array[left_index].bottom[0],
                            right_array[right_index].bottom[0]);
                y_min = min(left_array[left_index].bottom[1],
                            right_array[right_index].bottom[1]);
                z_min = min(left_array[left_index].bottom[2],
                            right_array[right_index].bottom[2]);

                x_max = max(left_array[left_index].top[0],
                            right_array[right_index].top[0]);
                y_max = max(left_array[left_index].top[1],
                            right_array[right_index].top[1]);
                z_max = max(left_array[left_index].top[2],
                            right_array[right_index].top[2]);

                nodes[index].bottom = {x_min, y_min, z_min};
                nodes[index].top = {x_max, y_max, z_max};
            }
            if (index == 0) {
                break;
            }
            index = nodes[index].parent;
        }

}

} // namespace gpu

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

    unsigned int prefix_length = bit_prefix(key_i, key_j);
    if (prefix_length == n_bits) {
        prefix_length += bit_prefix(i, j);
    }
    return prefix_length;
}

} // namespace grace
