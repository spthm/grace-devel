from operator import itemgetter

import numpy as np

import morton_keys
import bits
from Nodes import Node, LeafNode, AABB

class BinRadixTree(object):
    """Represents binary radix tree.  Contains methods for tree construction.

    Instance variables:
        primitives --
            a list of the objects contained within the tree
        n_primitives --
            the number of primitives in the tree
        n_per_leaf --
            the maximum allowed number of primitives within a leaf
        n_nodes --
            the number of non-leaf nodes in the tree
        n_leaves --
            the number of leaf nodes in the tree
        keys --
            a list of the morton keys corresponding to each primitive
        nodes --
            a list of the Node objects which make up the tree
        leaves --
            a list of the LeafNode objects in the tree

    Methods:
        from_primitives(primitives)
        short_from_primitives(primitives, n_per_leaf)
        build()
        compact()
        find_AABBs()
        generate_keys()
        sort_primitives_by_keys()
        update_node(node_index, end_index, split_index)

    """
    def __init__(self, primitives, n_per_leaf=1):
        """
        Initialize a binary tree from a list of (x,y,z) primitive co-ordinates.

        Morton keys are generated for the primitives, memory is allocated for
        the tree, but it is not constructed and the keys are not sorted.

        """
        super(BinRadixTree, self).__init__()
        self.primitives = np.array(primitives)
        self.n_primitives = len(primitives)
        self.n_per_leaf = n_per_leaf
        self.n_nodes = self.n_primitives - 1
        self.n_leaves = self.n_primitives

        self.keys = self.generate_keys()

        self.nodes = [Node.null() for i in range(self.n_nodes)]
        self.leaves = [LeafNode.null() for i in range(self.n_leaves)]

    @classmethod
    def from_primitives(cls, primitives):
        """Build a binary tree from a list of (x,y,z) primitive co-ordinates.

        The same as the default constructor, but the keys and primitves are
        then sorted and the tree built.

        """
        print("Allocating arrays, generating keys...")
        tree = cls(primitives)
        print("Sorting keys...")
        tree.sort_primitives_by_keys()
        print("Building tree...")
        tree.build()
        print("Setting AABBs...")
        print("")
        tree.find_AABBs()
        return tree

    @classmethod
    def short_from_primitives(cls, primitives, n_per_leaf):
        """
        Build a short binary tree from (x,y,z) positions and a max leaf size.

        The same as from_primitives, but a maximum leaf size > 1 may be
        provided, and the tree is compacted after construction.

        """
        print("Allocating arrays, generating keys...")
        tree = cls(primitives, n_per_leaf)
        print("Sorting keys...")
        tree.sort_primitives_by_keys()
        print("Building short tree...")
        tree.build()
        print("Compacting tree...")
        tree.compact()
        print("Setting AABBs...")
        print("")
        tree.find_AABBs()
        return tree

    def build(self):
        """
        Build a binary radix tree from the list of primitives and their keys.
        """
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

            self.update_node(node_index, end_index, split_index)

    def compact(self):
        """Remove null nodes and leaves from a tree with n_per_leaf > 1."""
        if self.n_per_leaf == 1:
            raise ValueError("n_per_leaf == 1, cannot compact tree.")

        n_leaves = self._count_valid_nodes(self.leaves)

        # index_shifts[i] == number of nodes removed for j < i.
        leaf_index_shifts = self._inclusive_null_node_scan(self.leaves)

        self.nodes = self._remove_null_nodes(self.nodes)
        self.leaves = self._remove_null_nodes(self.leaves)

        if len(self.leaves) != n_leaves:
            # Pretty much has to work.
            err_msg = "Compaction failed.  n_leaves != len(self.leaves)."
            raise ValueError(err_msg)
        if len(self.nodes) != n_leaves - 1:
            # Might get this if there's a bug.
            raise ValueError("Compaction failed.  n_nodes != n_leaves - 1.")

        self._shift_indices(leaf_index_shifts)

    def find_AABBs(self):
        for leaf_node in self.leaves:
            leaf_node._AABB = self._get_leaf_AABB(leaf_node)
            current_node = leaf_node.parent
            while current_node is not None:
                self._update_AABB(current_node, leaf_node.AABB)
                current_node = current_node.parent

    def generate_keys(self):
        """Return a list of morton keys, one for each primitive."""
        # morton_key_3D expects (x, y, z) as arguments.
        # Spheres are stored as (x, y, z, r)
        keys = np.array([morton_keys.morton_key_3D(*prim[0:3])
                         for prim in self.primitives])
        return keys

    def sort_primitives_by_keys(self):
        """Sort the list of primitives by their morton keys.

        The list of keys is also sorted, so key[i] == morton_key(primitive[i])
        after sorting.

        """
        packed_list = zip(self.keys, self.primitives)
        # Sorts by first element (the keys).
        packed_list.sort(key=itemgetter(0))
        # Unpack and convert to lists (zip outputs a list of tuples).
        self.keys, self.primitives = [np.array(t) for t in zip(*packed_list)]

    def update_node(self, node_index, end_index, split_index):
        """
        Write index attribute of the node spanning [node_index,end_index],
        and assign its children.  This node is assigned as their parent.

        If the span range is less than the number of nodes per leaf, write a
        null node instead.

        """
        # For self.n_per_leaf == 1, this is always true.
        # (A node spans >= 2 keys.)
        if abs(end_index - node_index) + 1 > self.n_per_leaf:
            self._write_node(node_index, end_index, split_index)
        else:
            # This node will be processed as a leaf by its parent,
            # or is the would-be child of a would-be node.
            self._write_null_node(node_index)

    def _common_prefix(self, i, j):
        """Return the longest common prefix of keys at indices i, j."""
        if j < 0 or j > self.n_primitives-1:
            return -1

        key_i, key_j = self.keys[i], self.keys[j]

        prefix_length = bits.common_prefix(key_i, key_j)
        if prefix_length == 32:
            # Identical keys.  Increase length of prefix using key indices.
            prefix_length += bits.common_prefix(i, j)
        return prefix_length

    def _count_valid_nodes(self, array):
        """Return the number of non-null nodes in array."""
        return sum([1 for node in array if not node.is_null()])

    def _find_node_end(self, i, d):
        """Return the other end of the node starting at i, with direction d."""
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
        """
        Return the split index for a node spanning [i,j], with direction d.
        """
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

    def _get_leaf_AABB(self, leaf):
        """Return the AABB for the leaf."""
        primitive = self.primitives[leaf.start_index]
        pos, r = primitive[0:3], primitive[3]
        box = AABB(pos-r, pos+r)
        # In case we have n_per_leaf > 1, loop through primitives.
        for i in range(1, leaf.span):
            primitive = self.primitives[leaf.start_index+i]
            pos, r = primitive[0:3], primitive[3]
            box.bottom = np.minimum(pos-r, box.bottom)
            box.top = np.maximum(pos+r, box.top)
        return box

    def _inclusive_null_node_scan(self, array):
        """
        Return the cumulative sum of the number of null nodes in array.

        Sum is inclusive.

        """
        running_total = 0
        N = len(array)
        prefix_sums = np.zeros(N, dtype=np.int32)
        for i in xrange(N):
            if array[i].is_null():
                running_total += 1
            prefix_sums[i] = running_total
        return prefix_sums

    def _node_direction(self, i):
        """
        Return the direction of the node starting (+1) or ending (-1) at i.
        """
        return np.sign(self._common_prefix(i, i+1) -
                       self._common_prefix(i, i-1))

    def _remove_null_nodes(self, array):
        """
        Return all non-null nodes in array, preserving the original order.
        """
        return [node for node in array if not node.is_null()]

    def _shift_indices(self, leaf_shifts):
        """
        Find the new, shifted children for all nodes after compaction.

        leaf_shifts should be constructed such that:

            new_index = current_index - leaf_shifts[current_index]

        Though in the case that we have a right child which is not a node,
        the calculation is:

            new_index = current_index - leaf_shifts[current_index-1]

        """
        # NB: This loop checks that the new indices have been calculated
        # properly, and updates each node's index.
        # Since node.left/right/parent are direct references to objects,
        # they don't need to be fixed.
        for node in self.nodes:
            left_index = node.left.index
            new_index = left_index - leaf_shifts[left_index]
            if node.left.is_leaf():
                assert(node.left is self.leaves[new_index])
            else:
                assert(node.left is self.nodes[new_index])
            node.left._index = new_index

            right_index = node.right.index
            if node.right.is_leaf():
                new_index = right_index -leaf_shifts[right_index]
                assert(node.right is self.leaves[new_index])
            else:
                # NB: right_index-1 == left_index.
                # We use this since we are fixing a node index, not a leaf index.
                # Nodes are, conceptually, split between their left and right
                # children => right child shifts same distance as left.
                new_index = right_index - leaf_shifts[right_index-1]
                assert(node.right is self.nodes[new_index])
            node.right._index = new_index

    def _update_AABB(self, node, new_box):
        if node.AABB is None:
            # AABB constructor makes a copy of the lists, which prevents
            # a parent overwriting its children's AABB bottom/top.
            node._AABB = AABB(new_box.bottom, new_box.top)
        else:
            # Node AABB has already been set by some subset of its primitives.
            node.AABB.bottom = np.minimum(node.AABB.bottom, new_box.bottom)
            node.AABB.top = np.maximum(node.AABB.top, new_box.top)

    def _write_node(self, i, j, split_idx):
        """
        Write index, level and child references of the node spanning [i,j],
        splitting at split_idx, and assign it as the parent of its children.
        """
        # Output child nodes/leaves, and set parents/children where possible.
        # Leaves are processed only once, so we can explicity construct them
        # here.
        # Nodes are processed twice (first by their parent, then by
        # themselves), so we may only update their properties.
        this_node = self.nodes[i]
        this_node._index = i
        this_node._level = self._common_prefix(i, j)

        left_child_start = min(i,j)
        left_child_span = split_idx - left_child_start + 1
        right_child_start = left_child_start + left_child_span
        right_child_span = max(i,j) - split_idx

        if left_child_span <= self.n_per_leaf:
            left = LeafNode(split_idx, this_node,
                            start=left_child_start, span=left_child_span)
            self.leaves[split_idx] = left
            assert(left.start_index + left.span - 1 < len(self.primitives))
        else:
            left = self.nodes[split_idx]
            left.parent = this_node

        if right_child_span <= self.n_per_leaf:
            right = LeafNode(split_idx+1, this_node,
                             start=right_child_start, span=right_child_span)
            self.leaves[split_idx+1] = right
            assert(right.start_index + right.span - 1 < len(self.primitives))
        else:
            right = self.nodes[split_idx+1]
            right.parent = this_node

        this_node.left = left
        this_node.right = right

    def _write_null_node(self, i):
        """Write the node starting at i as a null node."""
        self.nodes[i] = Node.null()
