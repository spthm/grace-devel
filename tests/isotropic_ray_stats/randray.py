import numpy as np

import rays as r

def _random_sphere_point(return_norm=False):
    """
    Return a point from a uniform distribution within a unit sphere at (0, 0).

    If the argument is True, the point and its squared-magnitude are returned.
    Else, only the point is returned.
    """
    while True:
        vec = np.random.uniform(low=-1.0, high=1.0, size=3)
        N = sum(x * x for x in vec)
        if N <= 1.0:
            if return_norm:
                return vec, np.sqrt(N)
            else:
                return vec

def random_direction():
    """
    Return a random point on the surface of a unit sphere.

    Operates via rejection sampling. If the distribution is uniform within the
    sphere, in three dimensions, then reducing that distribution to two
    dimensions in a way which does not modify the directional properties of
    the distribution (i.e. by modifying only lengths) must be uniform in the
    reduced number of dimensions also.

    Practically speaking, the in-sphere distribution is clearly invariant under
    rotations (by definition, even). Projecting this distribution onto the
    surface of the sphere then gives as a two-dimensional, still uniform,
    distribution.
    """
    vec, N = _random_sphere_point(True)
    return vec / N

def random_point(low=0.0, high=1.0):
    """Return a uniform random point with all coordinates in [low, high)."""
    return np.random.uniform(low=low, high=high, size=3)

def n_random_directions(n):
    """Return n random direction vectors."""
    vecs = np.array([_random_sphere_point(False) for i in range(n)])
    # This is quite a bit faster than np.linalg.norm for large n.
    N = np.sqrt((vecs * vecs).sum(axis=1))
    return vecs / N[:,np.newaxis]

def n_random_rays(n, origin, length):
    dirs = n_random_directions(n)
    origins = np.array(origin) * np.ones_like(dirs)
    lengths = length * np.ones_like(dirs)
    return r.RayCollection(origins, dirs, lengths)

if __name__ == '__main__':
    p = random_point(-10.0, 20.0)
    assert p.shape == (3,)
    for c in p:
        assert c >= -10.0 and c <= 20.0

    q = random_point(-10.0, 20.0)
    assert p.shape == (3,)
    # In principle not actually an error, but...
    assert not all(p == q)
    print("PASSED random point tests")

    d = random_direction()
    assert d.shape == (3,)
    N = sum(x * x for x in d)
    assert np.isclose(N, 1.0)
    print("PASSED random direction test")

    ds = n_random_directions(10)
    assert ds.shape == (10, 3)
    assert np.allclose((ds*ds).sum(axis=1), 1.0)
    print("PASSED random directions test")
