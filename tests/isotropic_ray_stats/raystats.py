import math

import numpy as np

def great_circle_distance(p, q):
    """
    Return the great-circle distance between p and q on a unit sphere.

    p array-like (3D co-ordinates, or a list or array of 3D co-ordinates).
    q array-like (3D co-ordinates, or a list or array of 3D co-ordinates).
    """
    p = np.asarray(p)
    q = np.asarray(q)

    if p.ndim > 2 or q.ndim > 2:
        raise ValueError("p and q must be 1D or 2D arrays.")

    # This is required for np.dot().
    if p.ndim < q.ndim:
        p, q = q, p

    n = np.cross(p, q)
    # This is faster than np.linalg.norm for large-array n.
    # axis=-1, not axis=1, since n may be 1D or 2D.
    mag_n = np.sqrt((n * n).sum(axis=-1))
    dot_p = np.dot(p, q)
    angular_separation = np.arctan2(mag_n, dot_p)
    return angular_separation

def _n_within(scale, sample):
    """
    Return the number of points within a distance scale of each point.

    sample should contain 3D cartesian co-ordinates for points on the sphere.

    scale may be scalar or a 1D array.
    sample may be a length-3 1D array, or an (N, 3)-size 2D array.

    For scalar scale, a single value is returned.
    For array-like scale, an equal-length array is returned.
    """
    scale = np.asarray(scale, dtype=float)
    is_scalar = False
    if scale.ndim == 0:
        # Make a true 1D array, of one element.
        scale = scale[np.newaxis]
        is_scalar = True

    sample = np.asarray(sample)
    if sample.ndim > 2:
        raise ValueError("sample must be a 1D or 2D array")
    elif sample.ndim == 1 and length(sample) != 3:
        # This k function is only defined for points on the surface of a sphere.
        raise ValueError("sample must be an (array of) 3D co-ordinates")

    n = len(sample)
    # It is possible to compute the below without a loop, using only numpy
    # broadcasting (and [:,np.newaxis]); however, it frequently leads to
    # MemoryErrors and is actually slower (at least on a workstation).
    n_within = np.zeros_like(scale)
    for p in sample:
        ds = great_circle_distance(p, sample)
        # We convert the scales into a column vector so the comparison will
        # work in the case that 'scale' is an array of scales to test
        # against. 'scale' will generally be smaller than 'ds', so it is the
        # one we transpose.
        n_within += np.sum(ds < scale[:,np.newaxis], axis=1)

    # Either we counted zero, or we counted at least n!
    assert not np.any((n_within != 0) * (n_within < n))
    # Points should not count themselves, so we have overcounted by n.
    n_within = np.maximum(np.zeros_like(n_within), n_within - n)
    if is_scalar:
        n_within = n_within[0]
    return n_within


def k_function(scale, sample):
    """
    Return the value of Ripley's K-function for sample.

    Return K(scale) for the points in sample, where scale is the on-sphere
    distance scale of interest, assuming a unit sphere.

    sample should contain 3D cartesian co-ordinates for points on the sphere.

    scale may be scalar or a 1D array.
    sample may be a length-3 1D array, or an (N, 3)-size 2D array.

    For scalar scale, a single value is returned.
    For array-like scale, an equal-length array is returned.
    """
    sample = np.asarray(sample)
    if sample.ndim == 1:
        n = 1
    else:
        n = len(sample)
    n_within = _n_within(scale, sample)
    # Unbiased estimator is / n(n-1), not / n^2.
    return 4. * np.pi / (n * (n - 1)) * n_within

def k_csr(scale):
    """
    Return K function on the unit sphere under complete spatial randomness.

    Scale is measured on-sphere (i.e. a great-circle distance)

    scale may be scalar or array-like.
    """
    # For non-unit sphere, the equation is actually
    #   2 pi R^2 * (1 - cos(scale / R))
    return 2. * np.pi * (1. - np.cos(scale))

def rayleigh(sample):
    """Return the Rayleigh statistic for directions in sample.

    All direction vectors in sample should be normalized.
    same must be array-like, (N, 3)-shape with N >= 2.
    """
    sample = np.asarray(sample)
    n = len(sample)
    if n < 2:
        raise ValueError("Cannot compute Rayleigh statistic for < 2 vectors.")
    if sample.shape[-1] != 3:
        raise ValueError("Must provide array of legnth-3 arrays.")

    coord_sum = sample.sum(axis=0)
    R2 = (coord_sum * coord_sum).sum()
    return 3. * R2 / n

def rayleigh_csr():
    """
    Return the value of the null distribution of the Rayleigh statistic.
    """
    return 0.0

def An_Gn(sample):
    """Return An, Gn statistics for directions in sample.

    All direction vectors in sample must be normalized.
    sample must be array-like, (N, 3)-shape with N >= 2.
    """
    sample = np.asarray(sample)
    n = len(sample)
    if n < 2:
        raise ValueError("Cannot compute An, Gn statistics from < 2 vectors.")
    if sample.shape[-1] != 3:
        raise ValueError("Must provide array of length-3 arrays.")

    an_sum2 = 0
    gn_sum2 = 0

    for i in range(n):
        di = sample[i]
        for j in range(i+1, n):
            dj = sample[j]
            # Unit circle, R*theta = s => theta = s.
            theta_ij = great_circle_distance(di, dj)
            an_sum2 += theta_ij
            gn_sum2 += np.sin(theta_ij)

    an_part = []
    gn_part = []
    for i in range(n - 1):
        di = sample[i]
        djs = sample[i+1:]
        theta_ijs = great_circle_distance(di, djs)
        an_part.append(math.fsum(theta_ijs))
        gn_part.append(math.fsum(np.sin(theta_ijs)))
    an_sum = math.fsum(an_part)
    gn_sum = math.fsum(gn_part)

    assert abs(an_sum - an_sum2) <= 1e-8 + 1e-5 * abs(an_sum2)
    assert abs(gn_sum - gn_sum2) <= 1e-8 + 1e-5 * abs(gn_sum2)

    an = n - (4. / (n * np.pi)) * an_sum
    gn = n / 2. - (4. / (n * np.pi)) * gn_sum

    return an, gn

if __name__ == '__main__':
    import math

    # For the first scale, the region surrounding each point contains no other
    # points.
    #     N = 0.
    # For second scale, the region surrounding the central point includes all
    # other points. The regions surrounding all other points contain only one
    # point (the central point). (The distance connecting each point to the
    # center point is less than the distance between, e.g., 'bottom' and 'right'
    # by the spherical law of cosines.)
    #     N = 8.
    # For the third scale, the region surrounding the non-central points
    # includes all-but-one of the points.
    #     N = 16.
    # For the fourth scale, regions surrounding each point include all other
    # points.
    #     N = 20.
    delta = 0.1
    arc_delta = math.atan(delta)
    scales = [0.9999  * arc_delta,
              1.00001 * arc_delta,
              0.9999  * 2 * arc_delta,
              1.00001 * 2 * arc_delta]
    Ns = [0, 8, 16, 20]

    norm = math.sqrt(1 + delta * delta)
    sample = [[ 0.0,          1.0,         0.0],          # Center
              [ 0.0,          1.0 / norm,  delta / norm], # Top
              [ 0.0,          1.0 / norm, -delta / norm], # Bottom
              [-delta / norm, 1.0 / norm,  0.0],          # Left
              [ delta / norm, 1.0 / norm,  0.0]]          # Right

    Ks = k_function(scales, sample)
    m = len(sample)
    for k, nin in zip(Ks, Ns):
        actual =  4. * np.pi / (m * (m - 1)) * nin
        assert np.isclose(k, actual)
    print("PASSED Ripley-K test")

    sample = [[1.0,  0.0,  0.0],
              [0.0,  1.0,  0.0],
              [0.0,  0.0,  1.0],
              [-1.0, 0.0,  0.0],
              [0.0,  -1.0, 0.0],
              [0.0,  0.0, -1.0],
              # Above vectors cancel out for Rayleigh test.
              # (4, 4, 7, 9) and (1, 4, 8, 9) are Pythagorean
              # quadruples, hence normalized.
              [-4.0/9.0, -4.0/9.0,  7.0/9.0],
              [-1.0/9.0,  4.0/9.0, -8.0/9.0]]
    W = rayleigh(sample)
    assert np.isclose(W, 3.0 * (5*5 + 0 + 1) / 81. / len(sample))
    print("PASSED Rayleigh test")

    # Data B5 of "Statistical Analysis of Spherical Data", Fisher, Lewis and
    # Embleton, 1987, Cambridge University Press.
    decl = np.array([36.5, 44.8, 349.9, 179.7, 148.8, 160.5, 6.7 , 178.2, 48.4,
                     275.1, 7.3 , 162.0, 188.7, 159.0, 100.9, 150.0, 168.8,
                     274.6, 338.2, 115.3, 293.9, 21.9, 115.7, 342.3, 315.1,
                     137.6, 30.2, 43.8, 274.1, 349.6, 287.5, 92.9, 63.7, 341.1,
                     16.9, 270.5, 343.1, 141.7, 1.3 , 34.6, 271.9, 338.4, 2.4 ,
                     144.2, 254.5, 333.3, 240.9, 8.1, 240.2, 339.2, 42.4,
                     260.8])
    incl = np.array([-70.5, 65.6, -17.2, -3.3, -37.3, -62.0, -73.7, -74.5, 40.1,
                     -3.7, -40.2, 41.2, -16.5, -30.7,  0.9, 38.3, 46.8, -39.6,
                     -73.0, 25.3, -26.7, -17.4, 14.2, 24.4, 68.6, 19.6, 18.6,
                     -7.4, -51.4, 6.6, 0.1, -32.6, 47.4, 48.5, 82.6, 67.0, 37.3,
                     61.9, 57.6, 54.0, -13.9, 14.9, -37.3, 44.2, 10.6, 70.6,
                     59.0, -4.9, -13.5, -22.6, -8.8, 29.9])
    # In [0, 2pi)
    azimuths = np.radians(decl)
    # In [0, pi]
    polars = np.radians(incl) + math.pi / 2.
    sample = [[math.sin(p)*math.cos(a), math.sin(p)*math.sin(a), math.cos(p)]
              for p, a in zip(polars, azimuths)]

    An, Gn = An_Gn(sample)
    # Comparison value from above reference, Example 5.34, pp. 150.
    assert np.isclose(An + Gn, 1.516, rtol=1e-4)
    print("PASSED An, Gn test")
