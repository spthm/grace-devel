#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace grace {

template <typename PrimitiveType>
class CudaBVH
{
public:
    typedef PrimitiveType primitive_type;
    typedef typename thrust::device_vector<PrimitiveType> primitive_vector;
    typedef typename thrust::device_vector<detail::CudaNode> node_vector;
    typedef typename thrust::device_vector<detail::CudaLeaf> leaf_vector;

    // Allocates space only.
    GRACE_HOST CudaBVH(const size_t N_primitives,
                       const int max_per_leaf = 1) :
        max_per_leaf(max_per_leaf), root_index(-1)
    {
        primitives.reserve(N_primitives);
        initialize_reserve();
    }

    GRACE_HOST CudaBVH(const PrimitiveType* host_ptr, const size_t N_primitives,
                       const int max_per_leaf = 1) :
        primitives(host_ptr, host_ptr + N_primitives),
        max_per_leaf(max_per_leaf), root_index(-1)
    {
        initialize_reserve();
    }

    GRACE_HOST CudaBVH(const std::vector<PrimitiveType>& primitives,
                       const int max_per_leaf = 1) :
        primitives(primitives), max_per_leaf(max_per_leaf), root_index(-1)
    {
        initialize_reserve();
    }

    GRACE_HOST CudaBVH(const thrust::host_vector<PrimitiveType>& primitives,
                       const int max_per_leaf = 1) :
        primitives(primitives), max_per_leaf(max_per_leaf), root_index(-1)
    {
        initialize_reserve();
    }

    GRACE_HOST CudaBVH(const thrust::device_vector<PrimitiveType>& primitives,
                       const int max_per_leaf = 1) :
        primitives(primitives), max_per_leaf(max_per_leaf), root_index(-1)
    {
        initialize_reserve();
    }

    template <typename PrimitiveIter>
    GRACE_HOST CudaBVH(PrimitiveIter first, PrimitiveIter last,
                       const int max_per_leaf = 1) :
        primitives(first, last), max_per_leaf(max_per_leaf), root_index(-1)
    {
        initialize_reserve();
    }

    GRACE_HOST CudaBVH(const CudaBVH<PrimitiveIter>& other) :
        primitives(other.primitives), nodes(other.nodes), leaves(other.leaves),
        max_per_leaf(other.max_per_leaf), root_index(other.root_index) {}


    const primitive_vector& primitives() const
    {
        return primitives;
    }
    primitive_vector& primitives()
    {
        return primitives;
    }

    const node_vector& nodes() const
    {
        return nodes;
    }
    node_vector& nodes()
    {
        return nodes;
    }

    const leaf_vector& leaves() const
    {
        return leaves;
    }
    leaf_vector& leaves()
    {
        return leaves;
    }

private:

    primitive_vector primitives;
    node_vector nodes;
    leaf_vector leaves;

    const int max_per_leaf;
    int root_index;

    GRACE_HOST initialize_reserve()
    {
        size_t estimate = primitives.size();
        if (max_per_leaf > 1) {
            estimate = (size_t)(1.4 * (primitives.size() / max_per_leaf));
        }

        nodes.reserve(esimate);
        leaves.reserve(esimate);
    }
};

} //namespace grace
