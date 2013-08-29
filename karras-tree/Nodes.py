class AABB(object):
    def __init__(self, bottom, top):
        self.bottom = list(bottom)
        self.top = list(top)

class Node(object):
    def __init__(self, index, left, right, parent, level=0, AABB=None):
        super(Node, self).__init__()
        self._index = index
        self._left = left
        self._right = right
        self._parent = parent
        self._level = level
        self._AABB = AABB

    @classmethod
    def null(cls):
        return cls(-1, None, None, None)

    @property
    def index(self):
        return self._index

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

    @property
    def level(self):
        return self._level

    @property
    def AABB(self):
        return self._AABB

    def is_leaf(self):
        return False

    def is_null(self):
        return self._index == -1

class LeafNode(object):
    def __init__(self, index, parent_node,
                 start=None, span=1, bbox=None):
        super(LeafNode, self).__init__()
        self._index = index
        self._parent = parent_node
        if start is None:
            self._start_index = self.index
        else:
            self._start_index = start
        self._span = span
        self._AABB = bbox

    @classmethod
    def null(cls):
        return cls(-1, None)

    @property
    def index(self):
        return self._index

    @property
    def start_index(self):
        return self._start_index

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent_node):
        self._parent = parent_node

    @property
    def span(self):
        return self._span

    @property
    def AABB(self):
        return self._AABB

    def is_leaf(self):
        return True

    def is_null(self):
        return self._index == -1
