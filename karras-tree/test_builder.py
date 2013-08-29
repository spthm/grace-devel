import itertools

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from builder import BinRadixTree, LeafNode

N = 20

# Store spheres as (x, y, z, r).
spheres = np.array(np.random.rand(N,4), dtype=np.float32)
spheres[:,3] /= float(N)

#while True:
short_tree = BinRadixTree.short_from_primitives(spheres, 10)
binary_tree = BinRadixTree.from_primitives(spheres)

def plot_AABB(cuboid, ax, **kwargs):
    xx_yy_zz = zip(cuboid.bottom, cuboid.top)
    vertices = np.array(list(itertools.product(*xx_yy_zz)))

    for (start, end) in itertools.combinations(vertices, 2):
        # We only want to plot lines that are parallel to an axis.
        if sum(abs(end-start) > 0) == 1:
            ax.plot3D(*zip(start, end), **kwargs)

for tree in (binary_tree, short_tree):
    print
    print "Keys:"
    for key in tree.keys:
        print "{0:030b}".format(key)

    print
    print "Leaf indices:"
    for leaf in tree.leaves:
        print leaf.index

    print
    print "Node indices, and leaf indices if present:"
    for node in tree.nodes:
        print "Node index:", node.index
        left, right = node.left, node.right
        out = 'Leaves: '
        if isinstance(left, LeafNode):
            out += str(left.index)
        if isinstance(right, LeafNode):
            out += ', ' + str(right.index)
        print out
        print

    print "Node indices, child indices, and leaf indices if present:"
    for node in tree.nodes:
        print "Node index: %d" % (node.index, )
        left, right = node.left, node.right
        print "Children: %d, %d" %(left.index, right.index)
        leaves = tuple([str(child.index) if child.is_leaf() else ' '
                        for child in (left, right)])
        print "Leaves:   %s, %s" % leaves
        print

    print "Node AABBs:"
    for node in tree.nodes:
        print "Node index:", node.index
        print "Bottom:", node.AABB.bottom
        print "Top:   ", node.AABB.top
        print

    print "Leaf AABBs:"
    for leaf in tree.leaves:
        print "Leaf index:", leaf.index
        print "Bottom:", leaf.AABB.bottom
        print "Top:   ", leaf.AABB.top
        print

    print "N nodes: ", len(tree.nodes)
    print "N leaves:", len(tree.leaves)

    # Set up the figure.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(1.0)
    ax.set_ylim(1.0)
    ax.set_zlim(1.0)

    # Plot the AABBs.
    colours = [mpl.cm.jet(1.*i/31.) for i in range(31)]
    for node in tree.nodes:
        plot_AABB(node.AABB, ax, color=colours[node.level%30])
    for leaf in tree.leaves:
        plot_AABB(leaf.AABB, ax, color=colours[30], alpha=0.5)

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
    xs, ys, zs = [np.concatenate((arr, arr.T)) for arr in (xs, ys, zs)]

    centres = tree.primitives[:,:3]
    radii = tree.primitives[:,3]
    for r, centre in zip(radii, centres):
        sphere_xs = r*xs + centre[0]
        sphere_ys = r*ys + centre[1]
        sphere_zs = r*zs + centre[2]
        ax.plot3D(np.ravel(sphere_xs), np.ravel(sphere_ys),
                  np.ravel(sphere_zs), color=colours[30], alpha=0.5)

plt.show()
