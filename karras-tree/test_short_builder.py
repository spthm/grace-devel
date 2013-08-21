import numpy as np

from builder import BinRadixTree, LeafNode

N = 1000

# Store spheres as (x, y, z, r).
spheres = np.array(np.random.rand(N,4), dtype=np.float32)
spheres[:,3] /= float(N)

binary_tree = BinRadixTree.from_primitives(spheres)
short_tree = BinRadixTree.short_from_primitives(spheres, 10)

print "Keys:"
for (key1, key2) in zip(binary_tree.keys, short_tree.keys):
    print "{0:030b}".format(key1) + " " + "{0:030b}".format(key2)

print
print "Leaf indices:"
for (leaf1, leaf2) in zip(binary_tree.leaves, short_tree.leaves):
    print leaf1.index, leaf2.index

print
print "Node indices, and leaf indices if present:"
for (node1, node2) in zip(binary_tree.nodes, short_tree.nodes):
    print "Node indices:", node1.index, node2.index
    left1, right1 = node1.left, node1.right
    left2, right2 = node2.left, node2.right
    out = "Leaves: "
    if isinstance(left1, LeafNode):
        out += str(left1.index)
    if isinstance(right1, LeafNode):
        out += ', ' + str(right1.index)
    out += "   Leaves: "
    if isinstance(left2, LeafNode):
        out += str(left2.index)
    if isinstance(right2, LeafNode):
        out += ', ' + str(right2.index)

    print out

print
print len(binary_tree.nodes), len(short_tree.nodes)
