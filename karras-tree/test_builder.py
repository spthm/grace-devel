import numpy as np

from builder import BinRadixTree, LeafNode

N = 100

# Store spheres as (x, y, z, r).
spheres = np.array(np.random.rand(N,4), dtype=np.float32)
spheres[:,3] /= float(N)

binary_tree = BinRadixTree.from_primitives(spheres)

print "Keys:"
for key in binary_tree.keys:
    print "{0:030b}".format(key)

print
print "Leaf indices:"
for leaf in binary_tree.leaves:
    print leaf.index

print
print "Node indices and leaf indices, if present:"
for node in binary_tree.nodes:
    print node.index
    left, right = node.left, node.right
    out = 'Leaves: '
    if isinstance(left, LeafNode):
        out += str(left.index)
    if isinstance(right, LeafNode):
        out += ', ' + str(right.index)
    print out
