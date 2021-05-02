from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt

def is_valid_line(line):
    return not line.startswith('#')

def next_valid_line(f):
    line = f.readline()
    while not is_valid_line(line):
        line = f.readline()
    return line

def surface_area(bottom, top):
    if len(bottom) != len(top):
        raise ValueError("Bottom and top must be of equal size.")

    Ls = [t - b for (t, b) in zip(top, bottom)]
    return np.prod(np.asarray(Ls))

fname = 'leaf_sizes.txt'
if len(sys.argv) > 1:
    fname = sys.argv[1]

sizes = []
surface_areas = []
with open(fname) as f:
    L = float(next_valid_line(f))
    N = float(next_valid_line(f))
    max_per_leaf = int(next_valid_line(f))

    for line in f:
        if not is_valid_line(line):
            continue
        svalues = line.split(', ')
        size = int(svalues[0])
        bottom = [float(s) for s in svalues[1:4]]
        top = [float(s) for s in svalues[4:]]
        SA = surface_area(bottom, top)
        sizes.append(size)
        surface_areas.append(SA)

sizes = np.array(sizes)
if sizes.sum() != N:
    print("Error! Total primitives: %d, total in leaves: %d" % (N, sizes.sum()))

box_area = surface_area([0, 0, 0], [L, L, L])
surface_areas = np.array(surface_areas)
total_area = surface_areas.sum()
print("Total leaf surface area fraction: %.3f" % (total_area / box_area))

surface_areas /= box_area

size_range = (1, max_per_leaf + 1)
size_hist, size_edges = np.histogram(sizes, bins=max_per_leaf,
                                     range=size_range)
SA_hist, SA_edges = np.histogram(surface_areas, bins=500)

plt.bar(size_edges[:-1], size_hist, width=1, bottom=0, align='center')
plt.xticks(range(*size_range), range(*size_range))

plt.figure()
plt.bar(SA_edges[:-1], SA_hist)

plt.show()
