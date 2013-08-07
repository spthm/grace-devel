import numpy as np
import morton_keys

class BinRadixTreeBuilder(object):
    """docstring for BinRadixTreeBuilder"""
    def __init__(self, primitives):
        super(BinRadixTree, self).__init__()
        self.primitives = primitives
        self.n_primitives = len(primitives)
        self.keys = morton_keys.generate_keys(self.primitives)
        self.sort(primitives, keys)
        self.build()

    def common_prefix(self, i, j):
        if j < 0:
            return -1
        if j > self.n_primitives - 1:
            return -1
        xor = self.keys[i] ^ self.keys[j]
        if xor > 0:
            # Count leading zeros of the xor.
            return int(31 - np.floor(np.log2(xor)))
        else: # idential keys
            xor = i ^ j
            # Increase length of prefix using key indices.
            return 32 + int(31 - np.floor(np.log2(xor)))


    def build(self):
        # This loop can be done in parallel.
        for node_idx in range(self.n_primitives-1):
            i = node_idx
            # Get the 'direction' of the node.
            # d = +1 => node begins at i (right branch).
            # d = -1 => node ends at i (left branch).
            d = np.sign(self.common_prefix(i, i+1) - self.common_prefix(i, i-1))

            # The lower bound for this node's prefix length comes from the
            # difference between its key and the key of its sibling.  (All keys
            # within this node must have a longer common prefix than this.)
            min_common_prefix = self.common_prefix(i, i-d)

            # Find an upper bound on this node's range of keys.
            l_max = 2
            while self.common_prefix(i, i + l_max*d) > min_common_prefix:
                # Since nodes are grouped into pairs, we don't have to search
                # through each value of l_max, only one for each possible decendant.
                l_max *= 2

            # Perform a binary search in [0, l_max-1] for the range end, j,
            # making use of the fact that everything in this node has a common
            # prefix longer than the prefix common to this node and its sibling.
            l = 0
            t = l_max / 2
            while t >= 1:
                if self.common_prefix(i, i + (l+t)*d) > min_common_prefix:
                    l += t
                t /= 2
            j = i +l*d

            # Perform a binary seach in [i,j] for the split position,
            # making use of the fact that everything before the split has a
            # longer common prefix than the node's prefix.
            node_prefix = self.common_prefix(i, j)
            s = 0
            t = l / 2
            while t >= 1:
                if self.common_prefix(i, i + (s+t)*d) > node_prefix:
                    s += t
                t /= 2
            split_idx = i + s*d + min(d,0)

            # Output child nodes/leaves.
            # if min(i,j) == split_idx:
            #     left = leaves[split_idx]
            # else:
            #     left = nodes[split_idx]
            # if max(i,j) == split_idx + 1:
            #     right = leaves[split_idx+1]
            # else:
            #     right = nodes[spit_idx+1]
            if abs(j - i) >=
            nodes[i] = (left, right)
