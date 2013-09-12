#include "BinaryRadixTree.h"
#include "nodes.h"

using namespace std;

namespace grace {

/** @brief A Binary Radix Tree constructor.
 *
 * Takes the positions of the primitives (currently only spheres) and
 * their number and allocates memory for the nodes and leaves of the tree.
 *
 * @param primitives A pointer to the the positions of the primitives.
 * @param nPrimitives The number of primitives.
 *
 */
template <typename Integer, typename Float>
BinaryRadixTree::BinaryRadixTree(const Float *primitives, const int nPrimitives) :
  primitives_(primitives),
  nPrimitives_(nPrimitives),
  nNodes_(nPrimitives-1),
  nLeaves_(nPrimitives)
{
    nodes_.resize(nPrimitives-1);
    leaves_.resize(nPrimitives);
    //TODO: Thrust copy *primitives to primitives_.
}

BinaryRadixTree::~BinaryRadixTree();

/** @brief Builds a binary radix tree.
 *
 * Calculates the morton key of each primitive and uses them to build a binary
 * radix tree on the GPU.
 */
template <typename Integer>
void BinaryRadixTree::build(void) {
    thrust::device_vector<Integer> keys = generateKeys();
    sortPrimitivesByKeys(keys);
    buildNodes(keys);
    findAABBs();
}

} // namespace grace
