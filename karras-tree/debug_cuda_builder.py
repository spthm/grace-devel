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


binary_tree = BinRadixTree(spheres)

for (i, key) in enumerate(binary_tree.keys):
    cuda_key = unsorted_keys[i]
    if key != cuda_key:
        print "Python unsorted key[%d] != CUDA unsorted key[%d]." %(i, i)
        print "{0:032b}".key + " != {0:032b}".format(cuda_key)

binary_tree.sort_primitives_by_keys()
for (i, key) in enumerate(binary_tree.keys):
    cuda_key = sorted_keys[i]
    if key != cuda_key:
        print "Python sorted key[%d] != CUDA sorted key[%d]." %(i, i)
        print "{0:032b}".key + " != {0:032b}".format(cuda_key)

binary_tree.build()
for (i, node) in enumerate(binary_tree.nodes):
    pass

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


