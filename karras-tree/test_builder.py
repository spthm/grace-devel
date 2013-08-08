import numpy as np

from builder import BinRadixTree

N = 100

# Store spheres as (x, y, z, r).
spheres = np.array(np.random.rand(N,4), dtype=np.float32)
spheres[:,3] /= float(N)

binary_tree = BinRadixTree.from_primitives(spheres)

for leaf in binary_tree.leaves:
    print leaf.index

for node in binary_tree.nodes:
    print node.index
