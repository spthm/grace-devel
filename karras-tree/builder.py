import numpy as np
import morton_keys

class Node(object):
    def __init__(self, left, right, parent):
        super(Node, self).__init__()
        self._left = left
        self._right = right
        self._parent = parent

    @classmethod
    def empty(cls):
        left, right, parent = ((None, None),)*3
        return cls(left, right, parent)

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left_child):
        self._left = left_child

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right_child):
        self._right = right_child

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent_node):
        self._parent = parent_node

    def is_leaf(self):
        return False

class LeafNode(object):
    def __init__(self, parent_node):
        super(LeafNode, self).__init__()
        self._parent = parent_node

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent_node):
        self._parent = parent_node

    def is_leaf(self):
        return True

class BinRadixTree(object):
    """docstring for BinRadixTree"""
    def __init__(self, primitives=[]):
        """
        Initialize BinRadixTree from a list of (x,y,z) primitive co-ordinates.

        Morton keys are generated for the primitives, memory is 'allocated' for
        the tree, but it is not constructed and the keys are not sorted.
        """
        super(BinRadixTree, self).__init__()
        self.primitives = primitives
        self.n_primitives = len(primitives)
        self.keys = self.generate_keys()
        self.nodes = [Node.empty() for i in range(self.n_primitives-1)]
        self.leaves = [LeadNode(None) for i in range(self.n_primitives)]

    @classmethod
    def from_primitives(cls, primitives):
        "Construct BinRadixTree from a list of (x,y,z) primitive co-ordinates."
        tree = cls(primitives)
        tree.sort_by_keys()
        tree.build()
        return tree

    def common_prefix(self, i, j):
        if j < 0:
            return -1
        if j > self.n_primitives - 1:
            return -1
        xor = self.keys[i] ^ self.keys[j]
        if xor > 0:
            # Count leading zeros of the xor.
            return 31 - int(np.floor(np.log2(xor)))
        else: # idential keys
            # Identical keys.  Increase length of prefix using key indices.
            xor = i ^ j
            index_prefix = 31 - int(np.floor(np.log2(xor)))
            return 32 + index_prefix

    def generate_keys(self):
        # morton_code expects (x, y, z) as arguments.
        self.keys = [morton_keys.morton_code(*pos) for pos in primitives]

    def sort_by_keys(self):
        packed_tuple = zip(self.keys, self.primitives)
        # Sorts by first element (the keys).
        packed_tuple.sort()
        # Unpack and *convert to list* (zip outputs tuples).
        self.keys, self.primitives = [list(t) for t in zip(*packed)]

    def build(self):
        if len(self.nodes != self.n_primitives-1):
            self.nodes = [Node.empty() for i in range(self.n_primitives-1)]
        if len(self.leaves != self.n_primitives):
            self.leaves = [LeadNode(None) for i in range(self.n_primitives)]
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
            j = i + l*d

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

            # Output child nodes/leaves, and set parents/children where possible.
            # Leaves are processed only once, so we can explicity construct them
            # here.
            # Nodes are processed twice (once by themselves, once by their
            # parent), so we may only update their properties.
            this_node = self.nodes[i]
            if min(i,j) == split_idx:
                left = self.leaves[split_idx] = LeafNode(this_node)
            else:
                left = self.nodes[split_idx]
                left.parent = this_node
            if max(i,j) == split_idx + 1:
                right = self.leaves[split_idx+1] = LeafNode(this_node)
            else:
                right = self.nodes[split_idx+1]
                right.parent = this_node
            this_node.left = left
            this_node.right = right
