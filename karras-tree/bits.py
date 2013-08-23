import numpy as np

def space_by_1(unspaced, order=10):
    """
    Places one zero between each (binary) digit of an integer.
    """
    spaced = 0
    for nth_bit in range(order):
        isolated_nth_bit = unspaced & (1 << nth_bit)
        spaced |= (isolated_nth_bit << nth_bit)
    return spaced

def space_by_2(unspaced, order=10):
    """
    Places two zeroes between each (binary) digit of an integer.
    """
    spaced = 0
    for nth_bit in range(order):
        isolated_nth_bit = unspaced & (1 << nth_bit)
        spaced |= (isolated_nth_bit << 2*nth_bit)
    return spaced

def map_to_int(value, order):
    """
    Converts a floating point value in [0,1] to an integer in [0, 2^order).
    """
    span = 2**int(order) - 1
    return int(value * span)

def common_prefix(a, b):
    """
    Finds the length of the longest common prefix of two integers a and b.
    """
    xor = a ^ b
    if xor > 0:
        # Count leading zeros of the xor.
        return 31 - int(np.floor(np.log2(xor)))
    else: # idential values
        return 32
