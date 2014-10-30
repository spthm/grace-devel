#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace grace {

/* -----------------------------------------------------------------------------
 * Nodes + Leaves -> Tree, then hierarchy + AABB -> nodes and indices -> leaves.
 * Then everything exept the unit tests, uniform_rays.cu, AABB_speed.cu and
 * zip_sort.cu needs fixing.
 * -----------------------------------------------------------------------------
 */
class Tree
{
public:
    // A 32-bit int allows for up to indices up to +2,147,483,647 > 1024^3.
    // Since we likely can't fit more than 1024^3 particles on the GPU, int ---
    // which is 32-bit on relevant platforms --- should be sufficient for the
    // indices.
    // We could use uint and set the root node to 1.  Then common_prefix_length
    // may return 0 for out-of-range results, rather than -1.
    //
    // nodes[4*node_ID + 0].x: left child index
    //                     .y: right child index
    //                     .z: parent index
    //                     .w: index of the leaf node farthest from
    //                         leaves[node_ID] which is still a descendant of
    //                         this node.
    // nodes[4*node_ID + 1].x = left_bx
    //                     .y = left_tx
    //                     .z = right_bx
    //                     .w = right_tx
    // nodes[4*node_ID + 2].x = left_by
    //                     .y = left_ty
    //                     .z = right_by
    //                     .w = right_ty
    // nodes[4*node_ID + 3].x = left_bz
    //                     .y = left_tz
    //                     .z = right_bz
    //                     .w = right_tz
    thrust::device_vector<int4> nodes;
    // Equal to the common prefix of the keys which this node spans.
    // Currently used only when verifying correct construction.
    thrust::device_vector<unsigned int> levels;
    // leaves[leaf_ID].x = index of first sphere
    //                .y = number of spheres in the leaf
    //                .z = parent index
    //                .w = padding
    thrust::device_vector<int4> leaves;

    Tree(size_t N_leaves) : nodes(4*(N_leaves-1)), leaves(N_leaves),
                                   levels(N_leaves-1) {}
};

class H_Tree
{
public:
    thrust::host_vector<int4> nodes;
    thrust::host_vector<unsigned int> levels;
    thrust::host_vector<int4> leaves;

    H_Tree(size_t N_leaves) : nodes(4*(N_leaves-1)), leaves(N_leaves),
                                     levels((N_leaves-1)) {}
};

} //namespace grace
