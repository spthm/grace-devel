#pragma once

namespace grace {

//TODO: Can this be included in the .cu file, for consistency?
/** @brief Represents a binary radix tree and contains methods for
 *         tree construction.
 */
template <typename UInteger, typename Float>
class BinaryRadixTree
{
    UInteger n_nodes_;
    UInteger n_leaves_;
    thrust::device_vector<Float> d_sphere_centres_;
    thrust::device_vector<Float> d_sphere_radii_;
    thrust::device_vector<Node> d_nodes_;
    thrust::device_vector<Leaf> d_leaves_;

public:
    BinaryRadixTree(const thrust::host_vector<Float> sphere_centres,
                    const thrust::host_vector<Float> sphere_radii);
    ~BinaryRadixTree();

    void build(void);

private:

};

} // namespace grace
