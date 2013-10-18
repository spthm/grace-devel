import numpy as np

import morton_keys
from builder import BinRadixTree, LeafNode

with open("../tests/indata/x_fdata.txt") as f:
  xdata = [float(line.strip()) for line in f]
with open("../tests/indata/y_fdata.txt") as f:
  ydata = [float(line.strip()) for line in f]
with open("../tests/indata/z_fdata.txt") as f:
  zdata = [float(line.strip()) for line in f]
with open("../tests/indata/r_fdata.txt") as f:
  rdata = [float(line.strip()) for line in f]

spheres = np.array(zip(xdata, ydata, zdata, rdata))

with open("../tests/outdata/unsorted_keys_base10.txt") as f:
  unsorted_keys = [int(line.strip()) for line in f]
with open("../tests/outdata/sorted_keys_base10.txt") as f:
  sorted_keys = [int(line.strip()) for line in f]

with open("../tests/outdata/nodes.txt") as f:
    pass
with open("../tests/outdata/leaves.txt") as f:
    pass

print "Number of x, y, z, r components: %d, %d, %d, %d" % (len(xdata),
                                                           len(ydata),
                                                           len(zdata),
                                                           len(rdata))
print "Number of unsorted keys:         %d" % (len(unsorted_keys), )
print "Number of sorted keys:           %d" % (len(sorted_keys), )


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


