import numpy as np

import bits

def morton_key_2D(x, y, order=10):
    if not isinstance(x, int):
        x, y, z = [bits.map_to_int(i, order) for i in (x,y)]
    lookup = [bits.space_by_1(i, order) for i in range(2**order)]
    # Mask higher-order bits.
    mask = 2**order - 1
    x &= mask
    y &= mask
    return lookup[y] << 1 | lookup[x]

def morton_key_3D(x, y, z, order=10):
    if not isinstance(x, int):
        x, y, z = [bits.map_to_int(i, order) for i in (x,y,x)]
    lookup = [bits.space_by_2(i, order) for i in range(2**order)]
    mask = 2**order - 1
    x &= mask
    y &= mask
    z &= mask
    return lookup[z] << 2 | lookup[y] << 1 | lookup[x]
