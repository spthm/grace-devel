import numpy as np

import morton_keys
from builder import BinRadixTree, LeafNode

spheres = np.array([
                   [0.691545367240906, 0.585628449916840, 0.869682788848877,
                    0.069154538214207],
                   [0.237461879849434, 0.522250175476074, 0.539603292942047,
                    0.023746188730001],
                   [0.275667190551758, 0.730803966522217, 0.638666570186615,
                    0.027566719800234],
                   [0.384435445070267, 0.082793198525906, 0.510645270347595,
                    0.038443546742201],
                   [0.951158463954926, 0.369021147489548, 0.019436636939645,
                    0.095115847885609],
                   [0.203072756528854, 0.525394618511200, 0.323507100343704,
                    0.020307276397943],
                   [0.295432031154633, 0.799987137317657, 0.177729815244675,
                    0.029543204233050],
                   [0.177435606718063, 0.994355678558350, 0.543971061706543,
                    0.017743561416864],
                   [0.676712870597839, 0.607463896274567, 0.888761699199677,
                    0.067671291530132],
                   [0.113813042640686, 0.869451284408569, 0.283733487129211,
                    0.011381304822862]
                  ])

keys = np.array([int('0b00111100001111100110100010011111', 2),
                 int('0b00110000001001101011100010011000', 2),
                 int('0b00110001110010010001111100011110', 2),
                 int('0b00100001001010000010101010100001', 2),
                 int('0b00001011001011010110011001100111', 2),
                 int('0b00010100001101000010111001101011', 2),
                 int('0b00010011100000111110001101011100', 2),
                 int('0b00110010011010111011110101000011', 2),
                 int('0b00111100101010011001110111000110', 2),
                 int('0b00010110000011111011010001100010', 2)])

binary_tree = BinRadixTree.from_primitives(spheres)

for i,key in enumerate(keys):
    if key not in binary_tree.keys:
        print "Python key", i, ":", "{0:032b}".format(binary_tree.keys[i])
        print "CUDA key  ", i, ":", "{0:032b}".format(key)

print "Nodes:"
for i in range(len(spheres)-1):
    print "i:              ", i
    print "left leaf flag: ", binary_tree.nodes[i].left.is_leaf()
    print "left:           ", binary_tree.nodes[i].left.index
    print "right leaf flag:", binary_tree.nodes[i].right.is_leaf()
    print "right:          ", binary_tree.nodes[i].right.index
    parent = binary_tree.nodes[i].parent
    if parent is None:
        parent = 0
    else:
        parent = parent.index
    print "parent:         ", parent
    print

print "Leaves:"
for i in range(len(spheres)):
    print "i:     ", i
    parent = binary_tree.leaves[i].parent
    if parent is None:
        parent = 0
    else:
        parent = parent.index
    print "parent:", parent
    print


