#pragma once

#include "grace/cuda/bvh.cuh"
#include "grace/detail/bvh_ref.h"

namespace grace {


//
// Public member functions
//

GRACE_HOST
void CudaBvh::from_host(const HostBvh& bvh)
{
    typedef detail::Bvh_const_ref<HostBvh> src_ref_type;
    src_ref_type src(bvh);

    thrust::host_vector<detail::CudaBvhNode> dst_nodes(bvh.num_nodes());
    thrust::host_vector<detail::CudaBvhLeaf> dst_leaves(bvh.num_leaves());

    const size_t num_nodes = bvh.num_nodes();
    for (size_t i = 0; i < num_nodes; ++i)
    {
        src_ref_type::node_type host_node = src.nodes()[i];
        int lidx = host_node.left_child();
        int ridx = host_node.right_child();

        AABBf lbox, rbox;
        if (host_node.left_is_inner(num_nodes))
            lbox = src.nodes()[lidx].aabb();
        else
            lbox = src.leaves()[lidx - num_nodes].aabb();

        if (host_node.right_is_inner(num_nodes))
            rbox = src.nodes()[ridx].aabb();
        else
            rbox = src.leaves()[ridx - num_nodes].aabb();

        detail::CudaBvhNode cuda_node(host_node.left_child(),
                                      host_node.right_child(),
                                      host_node.first_leaf(),
                                      host_node.first_leaf()
                                          + host_node.size() - 1);

        dst_nodes[i] = cuda_node;
    }

    const size_t num_leaves = bvh.num_leaves();
    for (size_t i = 0; i < num_leaves; ++i)
    {
        src_ref_type::leaf_type host_leaf = src.leaves()[i];

        detail::CudaBvhLeaf cuda_leaf(host_leaf.first_primitive(),
                                      host_leaf.size(),
                                      host_leaf.parent());

        dst_leaves[i] = cuda_leaf;
    }

    // Well. This is ugly. Surely there is a better way to hide the details of
    // our BVH classes?
    typedef detail::Bvh_ref<CudaBvh> dst_ref_type;
    dst_ref_type dst(*this);

    dst.nodes() = dst_nodes;
    dst.leaves() = dst_leaves;
}

GRACE_HOST
void CudaBvh::to_host(HostBvh& bvh) const
{
    typedef detail::Bvh_ref<HostBvh> dst_ref_type;
    typedef detail::Bvh_const_ref<CudaBvh> src_ref_type;
    dst_ref_type dst(bvh);
    src_ref_type src(*this);

    dst.nodes().resize(num_nodes());
    dst.leaves().resize(num_leaves());

    thrust::host_vector<detail::CudaBvhNode> src_nodes(src.nodes());
    thrust::host_vector<detail::CudaBvhLeaf> src_leaves(src.leaves());

    const size_t num_nodes = src_nodes.size();
    for (size_t i = 0; i < num_nodes; ++i)
    {
        detail::CudaBvhNode cuda_node = src_nodes[i];

        dst_ref_type::node_type host_node(cuda_node.left_child(),
                                          cuda_node.right_child(),
                                          cuda_node.first_leaf(),
                                          cuda_node.size(),
                                          cuda_node.AABB());
        dst.nodes()[i] = host_node;

        int lidx = cuda_node.left_child();
        int ridx = cuda_node.right_child();
        if (cuda_node.left_is_leaf(num_nodes))
            dst.leaves()[lidx - num_nodes].set_aabb(cuda_node.left_AABB());
        if (cuda_node.right_is_leaf(num_nodes))
            dst.leaves()[ridx - num_nodes].set_aabb(cuda_node.right_AABB());
    }

    const size_t num_leaves = src_leaves.size();
    for (size_t i = 0; i < num_leaves; ++i)
    {
        detail::CudaBvhLeaf cuda_leaf = src_leaves[i];

        // We already wrote the AABBs --- do not overwrite the leaves, just
        // modify them.
        dst.leaves()[i].set_first_primitive(cuda_leaf.first_primitive());
        dst.leaves()[i].set_size(cuda_leaf.size());
        dst.leaves()[i].set_parent(cuda_leaf.parent());
    }

    bvh.set_root_index(root_index());
}


} // namespace grace
