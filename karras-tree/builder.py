import numpy as np
import morton_keys
import bits

class Node(object):
    def __init__(self, index, left, right, parent):
        super(Node, self).__init__()
        self.index = index
        self._left = left
        self._right = right
        self._parent = parent

    @classmethod
    def empty(cls):
        return cls(0, None, None, None)

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
    def __init__(self, index, parent_node, span=1):
        super(LeafNode, self).__init__()
        self.index = index
        self._parent = parent_node
        self._span = span

    @classmethod
    def empty(cls):
        return cls(0, None)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent_node):
        self._parent = parent_node

    @property
    def span(self):
        return self._span

    def is_leaf(self):
        return True

class BinRadixTree(object):
    """Represents a binary radix tree.  Contains methods for tree construction."""
    def __init__(self, primitives=[], n_per_leaf=1):
        """
        Initialize BinRadixTree from a list of (x,y,z) primitive co-ordinates.

        Morton keys are generated for the primitives, memory is 'allocated' for
        the tree, but it is not constructed and the keys are not sorted.
        """
        super(BinRadixTree, self).__init__()
        self.primitives = primitives
        self.n_primitives = len(primitives)
        self.n_per_leaf = n_per_leaf
        self.n_nodes = self.n_primitives - self.n_per_leaf
        self.n_leaves = self.n_nodes + 1

        self.keys = self.generate_keys()

        self.nodes = [Node.empty() for i in range(self.n_nodes)]
        self.leaves = [LeafNode.empty() for i in range(self.n_leaves)]

    @classmethod
    def from_primitives(cls, primitives):
        "Construct a BinRadixTree from a list of (x,y,z) primitive co-ordinates."
        print("Allocating arrays, generating keys...")
        tree = cls(primitives)
        print("Sorting keys...")
        tree.sort_primitives_by_keys()
        print("Building tree..")
        tree.build()
        return tree

    @classmethod
    def short_from_primitives(cls, primitives, n_per_leaf):
        "Construct a short BinRadixTree from (x,y,z) positions and a max leaf size. "
        print("Allocating arrays, generating keys...")
        tree = cls(primitives, n_per_leaf)
        print("Sorting keys...")
        tree.sort_primitives_by_keys()
        print("Building short tree..")
        tree.build_short()
        return tree

    def generate_keys(self):
        "Returns a list of morton keys, one for each primitive."
        # morton_key_3D expects (x, y, z) as arguments.
        # Spheres are stored as (x, y, z, r)
        return [morton_keys.morton_key_3D(*pos[0:3]) for pos in self.primitives]

    def sort_primitives_by_keys(self):
        """
        Sorts the list of primitives by their morton keys.

        The list of keys is also sorted, so key[i] == morton_key(primitive[i])
        after sorting.
        """
        packed_tuple = zip(self.keys, self.primitives)
        # Sorts by first element (the keys).
        packed_tuple.sort()
        # Unpack and *convert to list* (zip outputs tuples).
        self.keys, self.primitives = [list(t) for t in zip(*packed_tuple)]

    def build(self):
        "Builds a binary radix tree from the list of primitives and their keys."
        # This loop can be done in parallel.
        for node_index in range(self.n_nodes):
            # Get the 'direction' of the node.
            # d = +1 => node begins at i (right branch).
            # d = -1 => node ends at i (left branch).
            direction = self._node_direction(node_index)

            end_index = self._find_node_end(node_index, direction)

            split_index = self._find_split_index(node_index,
                                                 end_index,
                                                 direction)
            #self._update_node(node_index, end_index, split_index)
            # For self.n_per_leaf, this is always true.  (A node spans >= 2 keys.)
            if abs(end_index - node_index) + 1 > self.n_per_leaf:
                self._update_node(node_index, end_index, split_index)
            else:
                print abs(end_index - node_index)
                # This node will be processed as a leaf by its parent,
                # or is the would-be child of a would-be node.
                self._make_null_node(node_index)

        if self.n_per_leaf > 1:
            self._compact_tree()

    def _compact_tree(self):
        pass

    def _node_direction(self, i):
        return np.sign(self._common_prefix(i, i+1) -
                       self._common_prefix(i, i-1))

    def _common_prefix(self, i, j):
        "Returns the longest common prefix of keys at indices i, j."
        if j < 0 or j > self.n_primitives-1:
            return -1

        key_i, key_j = self.keys[i], self.keys[j]

        prefix_length = bits.common_prefix(key_i, key_j)
        if prefix_length == 32:
            # Identical keys.  Increase length of prefix using key indices.
            print "Identical keys!"
            prefix_length += bits.common_prefix(i, j)
        return prefix_length

    def _find_node_end(self, i, d):
        "Returns the other end of the node starting at i, with direction d."
        # The lower bound for this node's prefix length comes from the
        # difference between its key and the key of its sibling.  (All keys
        # within this node must have a longer common prefix than this.)
        min_common_prefix = self._common_prefix(i, i-d)

        # Find an upper bound on this node's range of keys.
        l_max = 2
        while self._common_prefix(i, i + l_max*d) > min_common_prefix:
            # Since nodes are grouped into pairs, we don't have to search
            # through each value of l_max, only one for each possible decendant.
            l_max *= 2

        # Perform a binary search in [0, l_max-1] for the range end, j,
        # making use of the fact that everything in this node has a common
        # prefix longer than the prefix common to this node and its sibling.
        l = 0
        t = l_max / 2
        while t >= 1:
            if self._common_prefix(i, i + (l+t)*d) > min_common_prefix:
                l += t
            t /= 2
        return i + l*d

    def _find_split_index(self, i, j, d):
        "Returns the split index for node starting at i and ending at j, with direction d."
        # Perform a binary seach in [i,j] for the split position,
        # making use of the fact that everything before the split has a
        # longer common prefix than the node's prefix.
        node_prefix = self._common_prefix(i, j)
        s = 0
        t = (j - i) * d
        while True:
            # t = ceil(l/2), ceil(l/4), ... (in case l is odd).
            t = (t+1) / 2
            if self._common_prefix(i, i + (s+t)*d) > node_prefix:
                s += t
            if t == 1:
                break
        # If d = -1, then we actually found split_idx + 1.
        return i + s*d + min(d,0)

    def _update_node(self, i, j, split_idx):
        # Output child nodes/leaves, and set parents/children where possible.
        # Leaves are processed only once, so we can explicity construct them
        # here.
        # Nodes are processed twice (first by their parent, then by
        # themselves), so we may only update their properties.
        this_node = self.nodes[i]
        this_node.index = i
        left_child_span = split_idx - min(i,j) + 1
        right_child_span = max(i,j) - split_idx

        if left_child_span <= self.n_per_leaf:
            left = self.leaves[split_idx] = LeafNode(split_idx, this_node,
                                                     left_child_span)
        else:
            left = self.nodes[split_idx]
            left.parent = this_node

        if right_child_span <= self.n_per_leaf:
            right = self.leaves[split_idx+1] = LeafNode(split_idx+1, this_node,
                                                        right_child_span)
        else:
            right = self.nodes[split_idx+1]
            right.parent = this_node

        this_node.left = left
        this_node.right = right

    def _make_null_node(self, i):
        self.nodes[i] = Node.empty()

