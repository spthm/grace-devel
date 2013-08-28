import numpy as np

from builder import BinRadixTree, LeafNode

N = 100

# Store spheres as (x, y, z, r).
spheres = np.array(np.random.rand(N,4), dtype=np.float32)
spheres[:,3] /= float(N)

#while True:
short_tree = BinRadixTree.short_from_primitives(spheres, 10)
binary_tree = BinRadixTree.from_primitives(spheres)

print
print "Keys: (short, full)"
for (key1, key2) in zip(short_tree.keys, binary_tree.keys):
    print "{0:030b}".format(key1) + " " + "{0:030b}".format(key2)

print
print "Leaf indices: (short, full)"
for (leaf1, leaf2) in zip(short_tree.leaves, binary_tree.leaves):
    print leaf1.index, leaf2.index

print
print "Node indices, child indices, and leaf indices if present:"
for (node1, node2) in zip(short_tree.nodes, binary_tree.nodes):
    print "Node indices (short, full): %d, %d" %(node1.index, node2.index)
    left1, right1 = node1.left, node1.right
    left2, right2 = node2.left, node2.right
    print "Children (short): %d, %d" %(left1.index, right1.index)
    short_leaves = tuple([child.index if child.is_leaf() else -1
                         for child in (left1, right1)])
    print "Leaves (short):   %d, %d" % short_leaves
    print "Children (full):  %d, %d" %(left2.index, right2.index)
    full_leaves = tuple([child.index if child.is_leaf() else -1
                        for child in (left2, right2)])
    print "Leaves (full):    %d, %d" % full_leaves
    print

print "N nodes (short, full): ", len(short_tree.nodes), len(binary_tree.nodes)
print "N leaves (short, full):", len(short_tree.leaves), len(binary_tree.leaves)
