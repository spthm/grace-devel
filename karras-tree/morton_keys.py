import numpy as np

import bits

def map_to_int(value, order):
    """Return a floating point value [0,1] mapped to an integer [0, 2^order).
    """
    span = 2**int(order) - 1
    return int(value * np.float32(span))

def morton_key_2D(x, y, order=10):
    """Return the Morton key of two integers x and y."""
    if not isinstance(x, int):
        x, y, z = [map_to_int(i, order) for i in (x,y)]
    lookup = [bits.space_by_1(i, order) for i in range(2**order)]
    # Mask higher-order bits.
    mask = 2**order - 1
    x &= mask
    y &= mask
    return lookup[y] << 1 | lookup[x]

def morton_key_3D(x, y, z, order=10):
    """Return the Morton key of three integers x, y and z."""
    if not isinstance(x, int):
        if not isinstance(x, np.int32):
            if not isinstance(x, np.int64):
                x, y, z = [map_to_int(i, order) for i in (x,y,z)]
    lookup = [bits.space_by_2(i, order) for i in range(2**order)]
    mask = 2**order - 1
    x &= mask
    y &= mask
    z &= mask
    return lookup[z] << 2 | lookup[y] << 1 | lookup[x]
