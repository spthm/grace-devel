#include "BinaryRadixTree.h"
#include "nodes.h"

namespace grace {

/** @brief Represents a binary radix tree and contains methods for
 *         tree construction.
 */
template <typename Integer, typename Float>
class BinaryRadixTree
{
    const Float *primitives_;
    const int nPrimitives_;
    const int nNodes_;
    const int nLeaves_;
    thrust::device_vector<Node> nodes_;
    thrust::device_vector<Leaf> leaves_;

public:
    /** @brief A Binary Radix Tree constructor.
     *
     * Takes the positions of the primitives (currently only spheres) and
     * their number and allocates memory for the nodes and leaves of the tree.
     *
     * @param primitives A pointer to the the positions of the primitives.
     * @param nPrimitives The number of primitives.
     *
     */
    BinaryRadixTree::BinaryRadixTree(const Float *primitives, const int nPrimitives) :
        primitives_(primitives),
        nPrimitives_(nPrimitives),
        nNodes_(nPrimitives-1),
        nLeaves_(nPrimitives)
    { }

    ~BinaryRadixTree();

    void build(void) {

    }

    /* data */
};

} // namespace grace
