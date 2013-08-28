import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from builder import BinRadixTree, LeafNode

N = 10

# Store spheres as (x, y, z, r).
spheres = np.array(np.random.rand(N,4), dtype=np.float32)
spheres[:,3] /= float(N)

binary_tree = BinRadixTree.from_primitives(spheres)

print
print "Keys:"
for key in binary_tree.keys:
    print "{0:030b}".format(key)

print
print "Leaf indices:"
for leaf in binary_tree.leaves:
    print leaf.index

print
print "Node indices, and leaf indices if present:"
for node in binary_tree.nodes:
    print "Node index:", node.index
    left, right = node.left, node.right
    out = 'Leaves: '
    if isinstance(left, LeafNode):
        out += str(left.index)
    if isinstance(right, LeafNode):
        out += ', ' + str(right.index)
    print out

print "Node indices, child indices, and leaf indices if present:"
for node in binary_tree.nodes:
    print "Node index: %d" % (node.index, )
    left, right = node.left, node.right
    print "Children: %d, %d" %(left.index, right.index)
    leaves = tuple([child.index if child.is_leaf() else -1
                    for child in (left, right)])
    print "Leaves:   %d, %d" % leaves
    print

print "Leaf AABBs:"
for leaf in binary_tree.leaves:
    print "Leaf index:", leaf.index
    print "Bottom:", leaf.AABB.bottom
    print "Top:   ", leaf.AABB.top
    print

print "N nodes: ", len(binary_tree.nodes)
print "N leaves:", len(binary_tree.leaves)


def plot_AABB(cuboid, ax, **kwargs):
    xx_yy_zz = zip(cuboid.bottom, cuboid.top)
    vertices = np.array(list(itertools.product(*xx_yy_zz)))

    for (start, end) in itertools.combinations(vertices, 2):
        # We only want to plot lines that are parallel to an axis.
        if sum(abs(end-start) > 0) == 1:
            ax.plot3D(*zip(start, end), **kwargs)

# Set up the figure.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the AABBs.
for node in binary_tree.nodes:
    plot_AABB(node.AABB, ax, color='k')
for leaf in binary_tree.leaves:
    plot_AABB(leaf.AABB, ax, color='r')

# Draw the spheres.
# u and v are parametric variables.
u = np.linspace(0, 2*np.pi, 10)
v = np.linspace(0, np.pi, 10)

# ax.plot3D gives best performance, *and* looks good with the fewest
# number of points.  To make things appear more sphereical from all viewing
# angles, we take the transpose as well (gives horiztonal 'rings').
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(len(u)), np.cos(v))
sphere_xs = np.concatenate((xs, xs.T))
sphere_ys = np.concatenate((ys, ys.T))
sphere_zs = np.concatenate((zs, zs.T))

centres = binary_tree.primitives[:,:3]
radii = binary_tree.primitives[:,3]
for r, centre in zip(radii, centres):
    x = r*sphere_xs + centre[0]
    y = r*sphere_ys + centre[1]
    z = r*sphere_zs + centre[2]
    ax.plot3D(np.ravel(x), np.ravel(y), np.ravel(z), color='b')

plt.show()
