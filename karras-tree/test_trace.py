import numpy as np

from builder import BinRadixTree
from trace import Ray, sphere_hit

N = 1000
N_rays = 80

# Generate rays from random directions in [-1, 1).
rays = [Ray(*xyz) for xyz in np.random.rand(N_rays, 3)]
# Sort rays by their Morton key.
rays.sort(key=lambda ray: ray.key)

# Store spheres as (x, y, z, r).
spheres = np.array(np.random.rand(N,4), dtype=np.float32)
#spheres[:,3] /= float(N)

binary_tree = BinRadixTree.from_primitives(spheres)


# Trace.
hit_counts = np.zeros(N_rays)
for ray_index in range(len(rays)):
    trace_stack = np.zeros(31, dtype=int)
    stack_index = 0

    node_index = 0
    is_leaf = False
    ray_hit_count = 0

    ray = rays[ray_index]
    while (stack_index >= 0):

        while not is_leaf:
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

        while (is_leaf):
            if sphere_hit(ray, *spheres[node_index]):
                ray_hit_count += 1

            node_index = trace_stack[stack_index]
            stack_index -= 1
            is_leaf = binary_tree.nodes[node_index].right.is_leaf()
            node_index = binary_tree.nodes[node_index].right.index

    hit_counts[ray_index] = ray_hit_count

print hit_counts
