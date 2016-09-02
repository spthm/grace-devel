#pragma once

// This class exists only to provide access to CudaBVH's private members
// without exposing them as part of the public API.
//
// DO NOT #include "CudaBVH.cuh" or #include "CudaBVH-inl.cuh".
//
// For simplicity, CudaBVH.cuh will include this file.
// Compilation should fail if this file is #include'd before CudaBVH.cuh.
// It makes no sense for this implementation-detail class to be used if
// CudaBVH is not also required.

namespace grace {

namespace detail {

class CudaBVH_ref
{
public:
    typedef CudaBVH BVH;
    typedef typename BVH::node_vector::value_type node_type;
    typedef typename BVH::leaf_vector::value_type leaf_type;
    typedef typename BVH::node_vector::iterator node_iterator;
    typedef typename BVH::leaf_vector::iterator leaf_iterator;

    GRACE_HOST CudaBVH_ref(BVH& cuda_bvh) :
        _nodes_ref(cuda_bvh._nodes), _leaves_ref(cuda_bvh._leaves) {}

    BVH::node_vector& nodes()
    {
        return _nodes_ref;
    }

    const BVH::node_vector& nodes() const
    {
        return _nodes_ref;
    }

    BVH::leaf_vector& leaves()
    {
        return _leaves_ref;
    }


    const BVH::leaf_vector& leaves() const
    {
        return _leaves_ref;
    }

private:
    BVH::node_vector& _nodes_ref;
    BVH::leaf_vector& _leaves_ref;
};

class CudaBVH_const_ref
{
public:
    typedef CudaBVH BVH;
    typedef typename BVH::node_vector::value_type node_type;
    typedef typename BVH::leaf_vector::value_type leaf_type;
    typedef typename BVH::node_vector::iterator node_iterator;
    typedef typename BVH::leaf_vector::iterator leaf_iterator;

    GRACE_HOST CudaBVH_const_ref(BVH& cuda_bvh) :
        _const_nodes_ref(cuda_bvh._nodes), _const_leaves_ref(cuda_bvh._leaves) {}

    GRACE_HOST CudaBVH_const_ref(const BVH& cuda_bvh) :
        _const_nodes_ref(cuda_bvh._nodes), _const_leaves_ref(cuda_bvh._leaves) {}

    BVH::node_vector& nodes()
    {
        return const_cast<BVH::node_vector&>(_const_nodes_ref);
    }

    const BVH::node_vector& nodes() const
    {
        return _const_nodes_ref;
    }

    BVH::leaf_vector& leaves()
    {
        return const_cast<BVH::leaf_vector&>(_const_leaves_ref);
    }


    const BVH::leaf_vector& leaves() const
    {
        return _const_leaves_ref;
    }

private:
    const BVH::node_vector& _const_nodes_ref;
    const BVH::leaf_vector& _const_leaves_ref;
};

} // namespace detail

} // namespace grace
