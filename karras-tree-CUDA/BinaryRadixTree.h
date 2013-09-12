#pragma once

namespace grace {

//TODO: Can this be included in the .cu file, for consistency?
/** @brief Represents a binary radix tree and contains methods for
 *         tree construction.
 */
template <typename Integer, typename Float>
class BinaryRadixTree
{
    const int nPrimitives_;
    const int nNodes_;
    const int nLeaves_;
    thrust::device_vector<Float> primitives_;
    thrust::device_vector<Node> nodes_;
    thrust::device_vector<Leaf> leaves_;

public:
    BinaryRadixTree(const Float *primitives, const int nPrimitives);
    ~BinaryRadixTree();
    void build(void);

private:
    thrust::device_vector<Integer> generateKeys_(void);

};

} // namespace grace
