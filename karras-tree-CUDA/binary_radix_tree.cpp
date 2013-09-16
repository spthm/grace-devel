#include "binary_radix_tree.h"
#include "nodes.h"

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
BinaryRadixTree::BinaryRadixTree(const thrust::host_vector<Float> sphere_centres,
                                 const thrust::host_vector<Float> sphere_radii)
{
    // TODO: Throw exception if sphere_radii.size() != sphere_centres.size().
    n_leaves_ = (UInteger) sphere_radii.size();
    n_nodes_ = n_leaves_ - 1;

    d_nodes_.resize(n_nodes_);
    d_leaves_.resize(n_leaves);
    d_sphere_centres_ = sphere_centres;
    d_sphere_radii_ = sphere_radii;
}

BinaryRadixTree::~BinaryRadixTree();

/** @brief Builds a binary radix tree.
 *
 * Calculates the morton key of each primitive and uses them to build a binary
 * radix tree on the GPU.
 */
template <typename UInteger>
void BinaryRadixTree::build(void) {
    thrust::device_vector<UInteger> d_keys = generate_keys();
    sort_primitives_by_keys(d_keys);
    build_nodes(d_keys);
    find_AABBs();
}

} // namespace grace
