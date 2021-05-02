from __future__ import print_function

import numpy as np

import randray as rr

def _move_to_end(a, move_map):
    """
    Return copy of a, with elements at each index in move_map moved to the end.

    Relative ordering of all elements not in move_map is preserved. Relative
    ordering of all elements at each index in move_map is preserved.

    a should be a 1D array, and move_map an array-like of indices into the
    array a.
    """
    n = len(a)
    m = len(move_map)
    others_mask = np.ones(n, dtype=np.bool)
    others_mask[move_map] = False

    new_a = np.ones_like(a)
    new_a[:n - m] = a[others_mask]
    new_a[n - m:] = a[move_map]

    return new_a

def unbiased_directions(n):
    return rr.n_random_directions(n)

def nonsymmetric_biased_directions(n, m, f):
    """
    Return n directions, with n*f, on average, biased into m hemispheres.

    All biased directions are reordered to appear at the end of the returned
    array. The index of the first biased direction, or, equivalently, the number
    of unbiased directions, is also returned.

    The m biased hemispheres are chosen at random, and the directions which are
    not biased are drawn from a uniform random distribution.

    A direction is selected to be biased with probability (100*2f)%, giving an
    average of nf biased directions. (For a true unbiased distribution, and
    assuming m == 1, we expect half of the n*2f randomly-selected directions to
    already lie in the biased hemisphere, and therefore only nf are biased, on
    average. The minimum number of biased directions is 0, the maximum is n*2f,
    and the mean is nf.)

    Due to the above, the maximum allowed value of f is 0.5.

    For m > 1, each bias hemisphere has an equal probability of being chosen as
    the bias for a biased direction.
    """
    if f > 0.5 or f < 0.0:
        raise ValueError("f must be in [0, 0.5]")

    unbiased = unbiased_directions(n)
    # Direction d in hemisphere with direction h := dot(d, h) >= 0.
    bias_dirs = unbiased_directions(m)

    # Find indices of rays to bias.
    bias_filter = np.random.uniform(0.0, 1.0, n)
    bias_idxs = np.where(bias_filter < 2. * f)[0]
    n_biased = len(bias_idxs)

    # Select a hemisphere for each biased direction.
    bias_dir_idxs = np.random.randint(0, m, n_biased)

    # Some of the selected rays will already lie in their bias hemisphere.
    # They are ignored. Rays not in their bias hemisphere must have their
    # direction reversed.
    # bias_trigger[i] = np.dot(unbiased[bias_idxs][i],
    #                          bias_dirs[bias_dir_idxs][i])
    bias_trigger = np.sum(unbiased[bias_idxs]*bias_dirs[bias_dir_idxs], axis=1)
    bias_idxs = bias_idxs[np.where(bias_trigger < 0.)]
    n_biased = len(bias_idxs)
    n_unbiased = n - n_biased
    unbiased[bias_idxs] *= -1.

    biased = _move_to_end(unbiased, bias_idxs)

    return biased, n_unbiased

def symmetric_biased_directions(n, m, f):
    """
    Return n directions, with n*f symmetrically biased into m hemispheres.

    All biased directions are reordered to appear at the end of the returned
    array. The index of the first biased direction, or, equivalently, the number
    of unbiased directions, is also returned.

    Symmetric here means that all directions, d, in a given bias hemisphere
    have an exact complement direction -d in the returned array. A (d, -d) pair
    is counted as two biased directions, i.e. if asking for f = 0.5, then on
    average there will be (n / 2) biased directions returned, (n / 4) of which
    are the symmetric complement of the other (n / 4). The number of biased
    directions is always even.

    The m biased hemispheres are chosen at random, and the directions which are
    not biased are drawn from a uniform random distribution.

    A direction is selected to be biased with probability (100*f)%, giving an
    average of nf/2. biased directions. Every such direction d then has its
    symmetric partner, -d, added to the set of biased directions. Hence, on
    average, a total of nf directions are biased. (See also documentation for
    nonsymmetric_biased_directions().)

    Due to the above, the maximum allowed value of f is 0.5.

    For m > 1, each bias hemisphere has an equal probability of being chosen as
    the bias for a biased direction.
    """
    # Generate nonsymmetric bias, but with half the number of biased directions
    # we actually want. Then make it symmetric, by adding each biased
    # direction's symmetric complement. This required wiping out some of the
    # unbiased directions from the nonsymmetric-bias array.
    if f > 0.5 or f < 0.0:
        raise ValueError("f must be in [0, 0.5]")
    nonsym_biased, n_unbiased = nonsymmetric_biased_directions(n, m, f/2.)
    n_biased = n - n_unbiased

    # We discard the [n_unbiased - n_biased:n_unbiased] unbiased directions in
    # nonsym_biased.
    unbiased = nonsym_biased[:n_unbiased - n_biased]
    biased = nonsym_biased[n_unbiased:]
    n_unbiased = len(unbiased)

    sym_biased = np.empty(n_unbiased + 2*n_biased, dtype=nonsym_biased.dtype)

    sym_biased[:n_unbiased] = unbiased
    sym_biased[n_unbiased:n_unbiased + n_biased] = biased
    sym_biased[n_unbiased + n_biased:] = -1.0 * biased

    return sym_biased, 2 * n_biased


if __name__ == '__main__':
    arr = np.arange(10)
    idxs = np.array([1, 5, 8])
    arr = _move_to_end(arr, idxs)
    assert np.all(arr == np.array([0, 2, 3, 4, 6, 7, 9, 1, 5, 8]))
    print("PASSED array indices-move test")

    dirs, n_unbiased = nonsymmetric_biased_directions(100, 1, 0.25)
    biased = dirs[n_unbiased:]
    assert np.allclose((dirs*dirs).sum(axis=1), 1.0)
    print("PASSED nonsymmetric unimodal normalization")

    # pairwise_dotp = np.dot(biased, biased.T)
    pairwise_dotp = np.dot(dirs, dirs.T)
    min_dotp = np.amin(pairwise_dotp)
    assert abs(min_dotp) <= np.pi
    print("PASSED nonsymmetric unimodal all in hemisphere")

