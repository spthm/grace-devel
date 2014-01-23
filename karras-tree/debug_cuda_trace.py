import numpy as np

from builder import BinRadixTree
from trace import Ray, sphere_hit

with open("spheredata.txt") as f:
    spheres = [[float(n) for n in line.split()] for line in f]
spheres = np.array(spheres)

with open("raydata.txt") as f:
    rays = [Ray(*[float(n) for n in line.split()]) for line in f]

with open("hitdata.txt") as f:
    hitdata = [float(line) for line in f]

# # Generate rays from random directions in [-1, 1).
# rays = [Ray(*xyz) for xyz in np.random.rand(100, 3)]
# # Sort rays by their Morton key.
# rays.sort(key=lambda ray: ray.key)

# # Store spheres as (x, y, z, r).
# spheres = np.array(np.random.rand(1000,4), dtype=np.float32)
# #spheres[:,3] /= float(N)

binary_tree = BinRadixTree.from_primitives(spheres)


# Trace.
hit_counts = np.zeros(len(rays))
for ray_index in range(len(rays)):
    trace_stack = np.zeros(31, dtype=int)
    stack_index = 0

    node_index = 0
    is_leaf = False
    ray_hit_count = 0

    ray = rays[ray_index]
    while (stack_index >= 0):

        if not is_leaf:
            if ray.hit(binary_tree.nodes[node_index].AABB):
                stack_index += 1
                trace_stack[stack_index] = node_index
                is_leaf = binary_tree.nodes[node_index].left.is_leaf()
                node_index = binary_tree.nodes[node_index].left.index
            else:
                node_index = trace_stack[stack_index]
                stack_index -= 1
                is_leaf = binary_tree.nodes[node_index].right.is_leaf()
                node_index = binary_tree.nodes[node_index].right.index

        if (is_leaf):
            if sphere_hit(ray, *spheres[node_index]):
                ray_hit_count += 1

            node_index = trace_stack[stack_index]
            stack_index -= 1
            is_leaf = binary_tree.nodes[node_index].right.is_leaf()
            node_index = binary_tree.nodes[node_index].right.index

    hit_counts[ray_index] = ray_hit_count

for i, n_py_hits in enumerate(hit_counts):
    if n_py_hits != 0:
        print "A Py hit"
        print
    if n_py_hits != hitdata[i]:
        print "Hit count mismatch at", i
        print "Py hits", n_py_hits, "vs", hitdata[i], "CUDA hits!"
        print
