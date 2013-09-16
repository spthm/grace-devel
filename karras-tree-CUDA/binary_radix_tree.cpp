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
template <typename UInteger, typename Float>
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
template <typename UInteger>
void BinaryRadixTree::build(void) {
    thrust::device_vector<UInteger> keys = generate_keys();
    sort_primitives_by_keys(keys);
    build_nodes(keys);
    find_AABBs();
}

} // namespace grace
