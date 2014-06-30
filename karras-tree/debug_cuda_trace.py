import numpy as np

from builder import BinRadixTree
from Nodes import Node
from trace import Ray

with open("../karras-tree-CUDA/tests/indata/spheredata.txt") as f:
    spheres = [[float(n) for n in line.split()] for line in f]

with open("../karras-tree-CUDA/tests/indata/raydata.txt") as f:
    rays = [Ray(*[float(n) for n in line.split()]) for line in f]

with open("../karras-tree-CUDA/tests/outdata/hitdata.txt") as f:
    hitdata = [float(line) for line in f]

binary_tree = BinRadixTree.from_primitives(spheres)


print "Tracing..."
hit_counts = np.zeros(len(rays))
for ray_index in range(len(rays)):
    trace_stack = [Node.null(), ]
    node = binary_tree.nodes[0]
    ray_hit_count = 0

    ray = rays[ray_index]
    while (not node.is_leaf() and not node.is_null()):
        while (not node.is_leaf() and not node.is_null()):
            if ray.AABB_hit(node.AABB):
                trace_stack.append(node.right)
                node = node.left
            else:
                node = trace_stack.pop()

        while (node.is_leaf() and not node.is_null()):
            for i in range(node.span):
                if ray.sphere_hit(*spheres[node.start+i]):
                    ray_hit_count += 1
            node = trace_stack.pop()

    hit_counts[ray_index] = ray_hit_count

error = False
print
for i, n_py_hits in enumerate(hit_counts):
    if n_py_hits != hitdata[i]:
        error = True
        print "Hit count mismatch at", i+1
        print "Py hits", n_py_hits, "vs", hitdata[i], "CUDA hits!"
        print

if not error:
    print "All hits match! (Total " + str(int(sum(hit_counts))) + ")"
    print
