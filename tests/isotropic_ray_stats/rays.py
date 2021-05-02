import re

import numpy as np

# Rays saved as ox:<float>oy:<float> ... dz:<float>length:<float>.
_float_pattern=r"[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?"
_ray_pattern_str = r"^ox:(?P<ox>%(p)s)oy:(?P<oy>%(p)s)oz:(?P<oz>%(p)s)dx:(?P<dx>%(p)s)dy:(?P<y>%(p)s)dz:(?P<dz>%(p)s)length:(?P<length>%(p)s)$" \
                   % {'p': _float_pattern}
_ray_pattern = re.compile(_ray_pattern_str)

class Ray(object):
    """A GRACE-type ray."""
    def __init__(self, origin, direction, length):
        super(Ray).__init__()
        self.origin = np.array(origin, dtype=np.float64)
        self.dir = np.array(direction, dtype=np.float64)
        self.length = length

class RayCollection(object):
    """A collection of GRACE-type rays."""
    def __init__(self, origins, directions, lengths):
        super(RayCollection, self).__init__()
        # We always want a copy of the provided input. Else it might get
        # modified without our knowledge.
        self._dirs = np.array(directions, dtype=np.float64)
        self._origins = np.array(origins, dtype=np.float64)
        self._lengths = np.array(lengths, dtype=np.float64)

    @classmethod
    def from_ray_list(cls, lst):
        origins = np.array([ray.origin for ray in lst])
        dirs = np.array([ray.dir for ray in lst])
        lengths = np.array([ray.length for ray in lst])
        return cls(origins, dirs, lengths)

    @property
    def dirs(self):
        return self._dirs

    @dirs.setter
    def dirs(self, array):
        self._dirs = np.array(array, dtype=np.float64)

    @property
    def lengths(self):
        return self._lengths

    @lengths.setter
    def lengths(self, array):
        self._lengths = np.array(array, dtype=np.float64)

    @property
    def origins(self):
        return self._origins

    @origins.setter
    def origins(self, array):
        self._origins = np.array(array, dtype=np.float64)

    def append(self, rhs):
        self += rhs

    def copy(self):
        return RayCollection(self._origins, self._dirs, self._lengths)

    def __iadd__(self, rhs):
        if not isinstance(rhs, RayCollection):
            return TypeError("Cannot add RayCollection to non RayCollection")
        self._assert_equal_sizes()
        rhs._assert_equal_sizes()
        self._origins = np.concatenate(self._origins, rhs._origins)
        self._dirs = np.concatenate(self._dirs, rhs._dirs)
        self._lengths = np.concatenate(self._lengths, rhs._lengths)
        return self

    def __add__(self, rhs):
        lhs = self.copy()
        lhs += rhs
        return lhs

    def __getitem__(self, i):
        return Ray(self._origins[i], self._dirs[i], self._lengths[i])

    def __setitem__(self, i, ray):
        self._origins[i] = ray.origin
        self._dirs[i] = ray.dir
        self._lengths[i] = ray.length

    def __len__(self):
        return len(self._dirs)

    def _equal_sizes(self):
        return (len(self._dirs) == len(self._origins) and
                len(self._dirs) == len(self._lengths))

    def _assert_equal_sizes(self):
        if not self._equal_sizes():
            raise RuntimeError("Unequal-size direction, origin, length arrays")


def read_rays(fname):
    rays = []
    with open(fname) as f:
        for i, line in enumerate(f):
            match = re.match(_ray_pattern, line)
            if not match:
                raise RuntimeError("File contains invalid line\n  %s:%d"
                                   % (fname, i))
            dct = match.groupdict()
            o = np.array([dct['ox'], dct['oy'], dct['oz']], dtype=np.float64)
            d = np.array([dct['dx'], dct['dy'], dct['dz']], dtype=np.float64)
            rays.append(Ray(o, d, dct['length']))

    return RayCollection.from_ray_list(rays)

