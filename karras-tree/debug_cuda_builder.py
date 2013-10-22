import numpy as np

import morton_keys
from builder import BinRadixTree, LeafNode

with open("../karras-tree-CUDA/tests/indata/x_fdata.txt") as f:
    xdata = [float(line.strip()) for line in f]
with open("../karras-tree-CUDA/tests/indata/y_fdata.txt") as f:
    ydata = [float(line.strip()) for line in f]
with open("../karras-tree-CUDA/tests/indata/z_fdata.txt") as f:
    zdata = [float(line.strip()) for line in f]
with open("../karras-tree-CUDA/tests/indata/r_fdata.txt") as f:
    rdata = [float(line.strip()) for line in f]

spheres = np.array(zip(xdata, ydata, zdata, rdata))

with open("../karras-tree-CUDA/tests/outdata/unsorted_keys_base10.txt") as f:
    unsorted_keys = [int(line.strip()) for line in f]
with open("../karras-tree-CUDA/tests/outdata/sorted_keys_base10.txt") as f:
    sorted_keys = [int(line.strip()) for line in f]

node_indices = []
levels = []
left_leaf_flags = []
left_indices = []
right_leaf_flags = []
right_indices = []
node_parent_indices = []
node_AABBs_bottom = []
node_AABBs_top = []
with open("../karras-tree-CUDA/tests/outdata/nodes.txt") as f:
    line = f.readline()
    while line:
        node_indices.append(int(line.split()[1]))
        levels.append(int(f.readline().split()[1]))
        left_leaf_flags.append(f.readline().split()[3] == "True")
        left_indices.append(int(f.readline().split()[1]))
        right_leaf_flags.append(f.readline().split()[3] == "True")
        right_indices.append(int(f.readline().split()[1]))
        node_parent_indices.append(int(f.readline().split()[1]))
        node_AABBs_bottom.append([float(float_string.rstrip(','))
                                  for float_string
                                  in f.readline().split()[1:]])
        node_AABBs_top.append([float(float_string.rstrip(','))
                               for float_string
                               in f.readline().split()[1:]])
        f.readline() # Blank
        line = f.readline() # Next index if it exists, otherwise blank.

leaf_indices = []
leaf_parent_indices = []
leaf_AABBs_bottom = []
leaf_AABBs_top = []
with open("../karras-tree-CUDA/tests/outdata/leaves.txt") as f:
    line = f.readline()
    while line:
        leaf_indices.append(int(line.split()[1]))
        leaf_parent_indices.append(int(f.readline().split()[1]))
        leaf_AABBs_bottom.append([float(float_string.rstrip(','))
                                  for float_string
                                  in f.readline().split()[1:]])
        leaf_AABBs_top.append([float(float_string.rstrip(','))
                               for float_string
                               in f.readline().split()[1:]])
        f.readline()
        line = f.readline()

print "Number of x, y, z, r components: %d, %d, %d, %d" % (len(xdata),
                                                           len(ydata),
                                                           len(zdata),
                                                           len(rdata))
print "Number of unsorted keys:         %d" % (len(unsorted_keys), )
print "Number of sorted keys:           %d" % (len(sorted_keys), )
print "Number of nodes:                 %d" % (len(node_indices), )
print "Number of leaves:                %d" % (len(leaf_indices), )
print


binary_tree = BinRadixTree(spheres)

for (i, key) in enumerate(binary_tree.keys):
    cuda_key = unsorted_keys[i]
    if key != cuda_key:
        print "Python unsorted key [%d] != CUDA unsorted key [%d]." %(i, i)
        print "{0:032b}".format(key) + " != {0:032b}.".format(cuda_key)

binary_tree.sort_primitives_by_keys()
for (i, key) in enumerate(binary_tree.keys):
    cuda_key = sorted_keys[i]
    if key != cuda_key:
        print "Python sorted key [%d] != CUDA sorted key [%d]." %(i, i)
        print "{0:032b}".format(key) + " != {0:032b}.".format(cuda_key)

binary_tree.build()
for (i, node) in enumerate(binary_tree.nodes):
    if node.index != node_indices[i]:
        print "Python node index [%d] != CUDA node index [%d]." %(i, i)
        print "%d != %d." %(node.index, node_indices[i])
        print

    if node.level != levels[i]:
        print "Python node level [%d] != CUDA node level [%d]." %(i, i)
        print "%d != %d." %(node.level, levels[i])
        print

    # Handle special case that the parent node is None (i.e. a null node,
    # and so node.parent.index is not valid).
    if node.parent is None:
        # We should be at the root node.
        if node_parent_indices[i] != 0:
            print "Python (root) node [%d].  CUDA node parent [%d] %d != 0." \
                %(i, node_parent_indices[i], i)
            print
    else:
        if node.parent.index != node_parent_indices[i]:
            print "Python parent index [%d] != CUDA parent index [%d]." %(i, i)
            print "%d != %d." %(node.parent, node_parent_indices[i])
            print

    if node.left.index != left_indices[i]:
        print "Python left index [%d] != CUDA left index [%d]." %(i, i)
        print "%d != %d." %(node.left.index, left_indices[i])
        print

    if node.right.index != right_indices[i]:
        print "Python right index [%d] != CUDA right index [%d]." %(i, i)
        print "%d != %d." %(node.right.index, right_indices[i])
        print

    if node.left.is_leaf() != left_leaf_flags[i]:
        print "Python left leaf flag [%d] != CUDA left leaf flag [%d]." %(i, i)
        print "%r != %r." %(node.left.is_leaf(), left_leaf_flags[i])
        print

    if node.right.is_leaf() != right_leaf_flags[i]:
        print "Python right leaf flag [%d] != CUDA right leaf flag [%d]." \
            %(i, i)
        print "%r != %r." %(node.right.is_leaf(), right_leaf_flags[i])
        print

for (i, leaf) in enumerate(binary_tree.leaves):
    if leaf.index != leaf_indices[i]:
        print "Python leaf index [%d] != CUDA leaf index [%d]." %(i, i)
        print "%d != %d." %(leaf.index, leaf_indices[i])
        print

    if leaf.parent.index != leaf_parent_indices[i]:
        print "Python leaf parent index [%d] != CUDA leaf parent index [%d]." \
            %(i, i)
        print "%d != %d." % (leaf.parent.index, leaf_parent_indices[i])

binary_tree.find_AABBs()
for (i, node) in enumerate(binary_tree.nodes):
    AABB_diffs = np.array(node.AABB.bottom) - np.array(node_AABBs_bottom[i])
    if max(abs(AABB_diffs)) > 1E-9:
        print "Python node bottom AABB [%d] != CUDA node bottom AABB [%d]." \
            %(i, i)
        print "%r != \n%r." %(list(node.AABB.bottom), node_AABBs_bottom[i])
        print

    AABB_diffs = np.array(node.AABB.top) - np.array(node_AABBs_top[i])
    if max(abs(AABB_diffs)) > 1E-9:
        print "Python node top AABB [%d] != CUDA node top AABB [%d]." \
            %(i, i)
        print "%r != \n%r." %(list(node.AABB.top), node_AABBs_top[i])
        print

for (i, leaf) in enumerate(binary_tree.leaves):
    AABB_diffs = np.array(leaf.AABB.bottom) - np.array(leaf_AABBs_bottom[i])
    if max(abs(AABB_diffs)) > 1E-9:
        print "Python leaf bottom AABB [%d] != CUDA leaf bottom AABB [%d]." \
            %(i, i)
        print "%r != \n%r." %(list(leaf.AABB.bottom), leaf_AABBs_bottom[i])
        print

    AABB_diffs = np.array(leaf.AABB.top) - np.array(leaf_AABBs_top[i])
    if max(abs(AABB_diffs)) > 1E-9:
        print "Python leaf top AABB [%d] != CUDA leaf top AABB [%d]." \
            %(i, i)
        print "%r != \n%r." %(list(leaf.AABB.top), leaf_AABBs_top[i])
        print


# print "Nodes:"
# for i in range(len(spheres)-1):
#     print "i:              ", i
#     print "left leaf flag: ", binary_tree.nodes[i].left.is_leaf()
#     print "left:           ", binary_tree.nodes[i].left.index
#     print "right leaf flag:", binary_tree.nodes[i].right.is_leaf()
#     print "right:          ", binary_tree.nodes[i].right.index
#     parent = binary_tree.nodes[i].parent
#     if parent is None:
#         parent = 0
#     else:
#         parent = parent.index
#     print "parent:         ", parent
#     print

# print "Leaves:"
# for i in range(len(spheres)):
#     print "i:     ", i
#     parent = binary_tree.leaves[i].parent
#     if parent is None:
#         parent = 0
#     else:
#         parent = parent.index
#     print "parent:", parent
#     print


