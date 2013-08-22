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
print "Node indices, and leaf indices if present:"
for node in binary_tree.nodes:
    print "Node index:", node.index
    left, right = node.left, node.right
    out = 'Leaves: '
    if isinstance(left, LeafNode):
        out += str(left.index)
    if isinstance(right, LeafNode):
        out += ', ' + str(right.index)
    print out

print "Node indices, child indices, and leaf indices if present:"
for node in binary_tree.nodes:
    print "Node index: %d" % (node.index, )
    left, right = node.left, node.right
    print "Children: %d, %d" %(left.index, right.index)
    leaves = tuple([child.index if child.is_leaf() else -1
                    for child in (left, right)])
    print "Leaves:   %d, %d" % leaves
    print

print "N nodes: ", len(binary_tree.nodes)
print "N leaves:", len(binary_tree.leaves)
