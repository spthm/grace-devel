from __future__ import print_function

import math

import numpy as np

from random import n_random_directions, random_direction_in_cone

def _number_unbiased(n, f, alpha):
    # Volume of cone of opening angle alpha / Volume of sphere:
    #     P := 0.5 * (1 - cos(alpha/2))
    # Now, for a cone with opening angle of alpha, from an unbiased sample of n
    # directions, we expect n * P of them to lie within the cone.
    # However, for a given bias fraction f, we actually want n * (f + P).
    # So, we generate some m < n unbiased directions, followed by (n - m)
    # biased directions. We fix m such that we expect n * (f + P) direction
    # vectors to lie in the bias direction; that is,
    #     m * P + (n - m) = n * (f + P),
    # hence,
    P = 0.5 * (1. - math.cos(alpha / 2.))
    m = n * (1. - f / (1. - P))
    return int(round(m))

def unbiased_directions(n):
    return n_random_directions(n)

def biased_direction(primary_dir, alpha):
    return random_direction_in_cone(primary_dir, alpha)

def nonsymmetric_biased_directions(n, m, f, alpha=np.pi/2.):
    """
    Return n directions with n*f, on average, biased into m cones.

    The opening (apex) angle of the cones is specified by alpha. It must be
    in (0, pi].

    All biased directions appear at the end of the returned array. The index of
    the first biased direction, or, equivalently, the number of unbiased
    directions, and the m bias direction(s), are also returned.

    The m bias-cone directions are chosen at random, and the directions which
    are not biased are drawn from a uniform random distribution.

    The bias fraction f is defined such that the returned array has, on average,
    (n*f) more directions pointed into (at least one of the) bias cone(s) than
    would be expected if all n directions were perfectly uniform,.

    For a given n, f and alpha, the number of unbiased directions returned is
    constant.

    For m > 1, each bias cone has an equal probability of being chosen as
    the bias for a biased direction.
    """
    if f > 0.5 or f < 0.0:
        raise ValueError("f must be in [0, 0.5]")
    if alpha > np.pi or alpha <= 0:
        raise ValueError("alpha must be in (0, pi]")

    n_ub = _number_unbiased(n, f, alpha)
    n_b = n - n_ub
    unbiased = unbiased_directions(n_ub)
    bias_dirs = unbiased_directions(m)

    # Select a cone for each biased direction.
    bias_dir_idxs = np.random.randint(0, m, n_b)
    # Why not.
    bias_dir_idxs.sort()
    biased = [biased_direction(bias_dirs[i], alpha) for i in bias_dir_idxs]

    if n_b > 0:
        combined = np.concatenate((unbiased, np.array(biased)), axis=0)
    else:
        combined = unbiased

    return combined, n_ub, bias_dirs

def symmetric_biased_directions(n, m, f, alpha=np.pi/2.):
    """
    Return n directions with n*f, on average, symmetrically biased into m cones.

    The opening (apex) angle of the cones is specified by alpha. It must be
    in (0, pi) (a value exactly equal to pi would result in net-zero bias).

    All biased directions appear at the end of the returned array. The index of
    the first biased direction, or, equivalently, the number of unbiased
    directions, and the m bias direction(s), are also returned.

    Symmetric here means that all biased directions, d, in a given bias cone
    have an exact complement direction -d in the returned array. A (d, -d) pair
    is counted as two biased directions, i.e. if asking for f = 0.5, then on
    average there will be (n / 2) biased directions returned, (n / 4) of which
    are the symmetric complement of the other (n / 4). The number of biased
    directions is always even.

    The m bias-cone directions are chosen at random, and the directions which
    are not biased are drawn from a uniform random distribution.

    For m > 1, each bias hemisphere has an equal probability of being chosen as
    the bias for a biased direction.
    """
    # Generate nonsymmetric bias, but with half the number of biased directions
    # we actually want. Then make it symmetric, by adding each biased
    # direction's symmetric complement. This requires wiping out some of the
    # unbiased directions from the nonsymmetric-bias array.
    if f > 0.5 or f < 0.0:
        raise ValueError("f must be in [0, 0.5]")
    if alpha >= np.pi or alpha <= 0:
        raise VaueError("alpha must be in (0, pi)")

    nonsym_biased, n_unbiased, bias_dirs = \
        nonsymmetric_biased_directions(n, m, f/2., alpha)
    n_biased = n - n_unbiased

    # We discard the [n_unbiased - n_biased:n_unbiased] unbiased directions in
    # nonsym_biased.
    unbiased = nonsym_biased[:n_unbiased - n_biased]
    biased = nonsym_biased[n_unbiased:]
    n_unbiased = len(unbiased)

    sym_biased = np.empty((n_unbiased + 2*n_biased, 3),
                          dtype=nonsym_biased.dtype)

    sym_biased[:n_unbiased] = unbiased
    sym_biased[n_unbiased:n_unbiased + n_biased] = biased
    sym_biased[n_unbiased + n_biased:] = -1.0 * biased

    return sym_biased, n_unbiased, bias_dirs


if __name__ == '__main__':
    dirs, n_unbiased, bias_dirs = nonsymmetric_biased_directions(100, 1, 0.25)
    bias_dir = bias_dirs[0]
    biased = dirs[n_unbiased:]
    assert np.allclose((dirs*dirs).sum(axis=1), 1.0)
    print("PASSED nonsymmetric unimodal normalization")

    assert np.all(np.dot(biased, bias_dir) >= 0)
    print("PASSED nonsymmetric unimodal all in cone")


    dirs, n_unbiased, bias_dirs = symmetric_biased_directions(100, 1, 0.25)
    bias_dir = bias_dirs[0]
    biased = dirs[n_unbiased:]
    assert np.allclose((dirs*dirs).sum(axis=1), 1.0)
    print("PASSED symmetric unimodal normalization")

    assert len(biased) % 2 == 0
    print("PASSED symmetic unimodal even number of biased directions")

    assert (np.sum(np.dot(biased, bias_dir) > 0) ==
            np.sum(np.dot(biased, bias_dir) < 0))
    print("PASSED symmetric unimodal half in each cone")


    alpha = np.pi / 2.
    dirs, n_unbiased, bias_dirs = nonsymmetric_biased_directions(100, 5, 0.25,
                                                                 alpha=alpha)
    biased = dirs[n_unbiased:]
    assert np.allclose((dirs*dirs).sum(axis=1), 1.0)
    print("PASSED nonsymmetric multimodal normalization")

    # 'bias direction' == the center direction of a cone into which some
    #                     directions are biased.
    # 'biased direction' == a direction which was deliberately chosen to point
    #                       into the cone of the above 'bias directions'
    # 'unbiased direction' == a direction drawn from a uniform random sample
    # We don't know which of the m = 5 bias directions each biased direction
    # corresponds to. So just compare against all of them, and note that we
    # must have dot(direction[i], bias_direction[j]) >= cos(alpha / 2) for at
    # least one value of j in [0, m = 5) for all values of i in [0, n_biased).
    # Replacing 'biased' with 'dirs' in np.dot() should ~always lead to failure.
    pairwise_dotp = np.dot(biased, bias_dirs.T)
    assert np.all(np.amax(pairwise_dotp, axis=1) >= np.cos(alpha / 2.))
    print("PASSED nonsymmetric multimodal all in hemispheres")


    dirs, n_unbiased, bias_dirs = symmetric_biased_directions(100, 3, 0.25)
    biased = dirs[n_unbiased:]
    assert np.allclose((dirs*dirs).sum(axis=1), 1.0)
    print("PASSED symmetric multimodal normalization")

    # We do not know which bias direction is associated with each biased
    # direction. So we perform the ~same test as the unimodal case --- that
    # every direction has a symmetric partner. We assume the probability of the
    # number of biased directions being symmetric wrt. all three axes without
    # each having an exact symmetric partner (i.e. by chance) to be low.
    # It appears to hold, in that replacing 'biased' with 'dirs' below ~always
    # results in failure (len(dirs) is even).
    sym_dirs = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
    assert np.all(np.sum(np.dot(biased, sym_dirs.T) > 0, axis=0) ==
                  np.sum(np.dot(biased, sym_dirs.T) < 0, axis=0))
    print("PASSED symmetric multimodal symmetry test")



