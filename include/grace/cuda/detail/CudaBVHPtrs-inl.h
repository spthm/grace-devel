#pragma once

//
// DO NOT #include "CudaBVH.h" or #include "CudaBVH-inl.h".
//
// For simplicity, CudaBVH.h will include this file.
// Compilation should fail if this file is #include'd before CudaBVH.h.
// It makes no sense for this implementation-detail class to be used if
// CudaBVH is not also required.
//

namespace grace {

namespace detail {

template <typename PrimitiveType>
class CudaBVHPtrs
{
public:
    typedef CudaBVH<PrimitiveType> BVH;
    typedef typename BVH::primitive_vector::value_type primitive_type;
    typedef typename BVH::node_vector::value_type node_type;
    typedef typename BVH::leaf_vector::value_type leaf_type;

    const primitive_type* primitives;
    const node_type* nodes;
    const leaf_type* leaves;
    const size_t N_primitives;
    const size_t N_nodes;
    const size_t N_leaves;

    GRACE_HOST CudaBVHPtrs(const BVH& cuda_bvh) :
        primitives( thrust::raw_pointer_cast(cuda_bvh.primitives().data())),
        nodes( thrust::raw_pointer_cast(cuda_bvh.nodes().data()) ),
        leaves( thrust::raw_pointer_cast(cuda_bvh.leaves().data()) ),
        N_primitives( cuda_bvh.primitives().size() ),
        N_nodes( cuda_bvh.nodes.size() ),
        N_leaves( cuda_bvh.leaves.size() ) {}
};

} // namespace detail

} // namespace grace
