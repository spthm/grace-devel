#include "Nodes.h"
#include "BinaryRadixTree_kernels.h"

namespace grace {

template <typename UInteger>
void build_nodes(thrust::device_vector<Node> d_nodes,
                 thrust::device_vector<Leaf> d_leaves,
                 thrust::device_vector<UInteger> d_keys)
{
    UInteger n_keys = d_keys.size();
    unsigned char n_bits_per_key = CHAR_BIT * sizeof(UInteger);

    build_nodes_kernel(thrust::raw_pointer_cast(d_nodes.data()),
                       thrust::raw_pointer_cast(d_leaves.data()),
                       thrust::raw_pointer_cast(d_keys.data()).
                       n_keys,
                       n_bits_per_key)

}

template <typename UInteger>
void find_AABBs(thrust::device_vector<Node> d_nodes,
                thrust::device_vector<Leaf> d_leaves,
                thrust::device_vector<float> d_sphere_centres,
                thrust::device_vector<float> d_sphere_radii)
{
    thrust::device_vector<unsigned char> d_AABB_flags;

    UInteger n_leaves = d_leaves.size();
    d_AABB_flags.resize(n_leaves);

    find_AABBs_kernel(thrust::raw_pointer_cast(d_nodes.data()),
                      thrust::raw_pointer_cast(d_leaves.data()),
                      n_leaves,
                      thrust::raw_pointer_cast(d_sphere_centres.data()),
                      thrust::raw_pointer_cast(d_sphere_radii.data()),
                      thrust::raw_pointer_cast(d_AABB_flags.data()) )
}


} // namespace grace
