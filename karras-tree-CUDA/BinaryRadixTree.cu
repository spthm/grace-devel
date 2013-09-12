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

namespace gpu {

template <typename Integer>
__global__ void BinaryRadixTree::buildNodesKernel(
  Integer *nodes,
  Integer *leaves,
  Integer *keys,
  nKeys) {
    int index;
    int direction;
    int end;
    int split;

    index = threadIdx.x + blockIdx.x * blockDim.x;

    direction = nodeDirection_(index, keys, nKeys);
    end = nodeEnd_(index, direction, keys, nKeys);
    split = nodeSplitIndex_(index, end, direction, keys, nKeys);

    updateNode(index, end, split);
}

template <typename Integer>
__device__ int commonPrefix_(int i, int j, Integer *keys, int N) {
    if (j < 0 || j > N)
        return -1;
    Integer key_i = keys[i];
    Integer key_j = keys[j];

    int prefixLength = bits::commonPrefix(key_i, key_j);
    if (prefixLength == 32)
        prefixLength += bits::commonPrefix(i, j);
    return prefixLength;
}

template <typename Integer>
__device__ int nodeDirection_(int nodeIndex, Integer *keys, int nKeys) {
    //TODO: Cuda sign function?
    return sign(commonPrefix_(nodeIndex, nodeIndex+1, keys, nKeys),
                commonPrefix_(nodeIndex, nodeIndex-1, keys, nKeys));
}

template <typename Integer>
__device__ int nodeEnd_(int nodeIndex, int nodeDirection, Integer *keys, int nKeys) {
    int minPrefix = commonPrefix_(nodeIndex, nodeIndex-nodeDirection, keys, nKeys);

    int lMax = 2;
    while (commonPrefix_(nodeIndex, nodeIndex + lMax*nodeDirection, keys, nKeys) > minPrefix)
        lMax *= 2;

    int l = 0;
    int t = lMax / 2;
    while (t >= 1) {
        if (commonPrefix_(nodeIndex, nodeIndex + (l+t)*nodeDirection, keys, nKeys) > minPrefix)
            l += t;
        t /= 2;
    return nodeIndex + l*nodeDirection;
    }
}

template <typename Integer>
__device__ int nodeSplitIndex_(nodeIndex, nodeEnd, nodeDirection, Integer *keys, int nKeys) {
    prefix = commonPrefix_(nodeIndex, nodeEnd, keys, nKeys);
    int s = 0;
    int t = (nodeEnd - nodeIndex) * nodeDirection;
    while true {
        t = (t+1) / 2;
        if (commonPrefix_(nodeIndex, nodeIndex + (s+t)*nodeDirection, keys, nKeys) > prefix)
            s += t;
        if t == 1
            break;
    return nodeIndex + s*nodeDirection + min(nodeDirection, 0);
    }
}

} // namespace gpu

} // namespace grace
