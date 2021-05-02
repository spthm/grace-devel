#!/usr/bin/python2

import argparse
from collections import OrderedDict
from functools import partial
from multiprocessing import Process, Pipe, cpu_count
from sys import stdout
from time import sleep
from warnings import warn

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# This MUST MATCH the N_rays in ripleyk_stats.cu.
_DIRS_PER_SAMPLE = 9600
# This MUST MATCH the Rs variable in ripleyk_stats.cu.
_test_scales = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25,
                np.pi / 2.0]
_CONFIDENCE = 0.95

def _UPPER(X):
    return lambda x: x >= X

def _LOWER(X):
    return lambda x: x <= X

def p_value(data, condition):
    """Return the p-value of the given condition from the given data.

    If condition does not hold for any element of data, the p-value will be
    non-zero but unreliable, likely overestimated.

    data must be array-like.
    condition must be callable for each element of data, and return int-like
    """
    count = sum(condition(x) for x in data)
    if count == 0:
        warn("Zero occurences of condition, p-value unreliable", RuntimeWarning)
    # We have +1s because the test statistic must be included in the reference
    # distribution; that is, the continuous reference distribution we are
    # attempting to emulate must have a single occurence of x == X.
    # See e.g.
    # https://www.biomedware.com/files/documentation/clusterseer/MCR/Calculating_Monte_Carlo_p-values.htm
    # Permutation P-values Should Never Be Zero: Calculating Exact P-values
    # When Permutations Are Randomly Drawn.
    # Gordon K. Smyth & Belinda Phipson (2011) pg 4:
    #   http://www.statsci.org/smyth/pubs/permp.pdf
    # On Estimating P Values by Monte Carlo Methods, Warren J. Ewens (2003),
    # Am J Hum Genet. 72(2):
    #   http://www.ncbi.nlm.nih.gov/pmc/articles/PMC379178/
    # Bootstrap Methods and their Application, A. C. Davison and D. V. Hinkley.
    # (1997), Cambridge University Press:
    #   http://dx.doi.org/10.1017/CBO9780511802843
    return float(count + 1) / (len(data) + 1)

def find_upper_limit(data, confidence=_CONFIDENCE):
    """Return X such that P(x >= X) <= 1 - confidence for x in data.

    In the case that X does not lie in data, the p-value will be non-zero but
    unreliable, likely overestimated, and may exceed 1 - confidence.
    """
    for X in np.linspace(np.min(data), np.max(data), num=min(100, len(data))):
        p = p_value(data, _UPPER(X))
        if p <= 1 - confidence:
            break
    return X

def find_lower_limit(data, confidence=_CONFIDENCE):
    """Return X such that P(x <= X) <= 1 - confidence for x in data.

    In the case that X does not lie in data, the p-value will be non-zero but
    unreliable, likely overestimated, and may exceed 1 - confidence.
    """
    for X in np.linspace(np.max(data), np.min(data), num=min(100, len(data))):
        p = p_value(data, _LOWER(X))
        if p <= 1 - confidence:
            break
    return X

def isotropic_sample(Ndirs=_DIRS_PER_SAMPLE):
    """Return Ndirs isotropic normalized direction vectors."""
    dirs = np.random.normal(size=(Ndirs, 3))
    N = np.sqrt((dirs * dirs).sum(axis=1))
    return dirs / N[np.newaxis].T

def great_circle_distance(p, q, R=1.0):
    """Return the great-circle (on-sphere) distance between p and q.

    R (optional) is the radius of the circle.

    R must be scalar.
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

    mcrossp = np.linalg.norm(np.cross(p, q), axis=-1)
    dotp = np.dot(p, q)
    angular_separation = np.arctan2(mcrossp, dotp)
    return R * angular_separation

def k_function(scale, sample):
    """Return Ripley's K function on the surface of the unit sphere.

    Return K(scale) for the points in sample, where scale is the on-sphere
    distance scale of interest.

    sample should contain 3D cartesian co-ordinates for points on the sphere.

    scale may be scalar or a 1D array of scales.
    sample may be a length-3 1D array, or an (N, 3) sized 2D array.
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

    N = len(sample)
    # It is possible to compute the below without a loop, using only numpy
    # broadcasting (and [np.newaxis], .T); however, it frequently leads to
    # MemoryErrors and is actually slower (at least on a workstation).
    N_within = np.zeros_like(scale)
    for p in sample:
        ds = great_circle_distance(p, sample)
        # We convert the scales into a column vector so the comparison will
        # work in the case that 'scale' is an array of scales to test
        # against. 'scale' will generally be smaller than 'ds', so it is the
        # one we transpose.
        N_within += np.sum(ds < scale[np.newaxis].T, axis=1)

    # Points should not count themselves, so we have overcounted by N.
    N_within -= N
    if is_scalar:
        N_within = N_within[0]
    return 4 * np.pi / (N * (N - 1)) * N_within

def CSR(scale):
    """Return K function on the unit sphere under Complete Spatial Randomness.

    scale may be scalar or array-like.
    """
    return 2 * np.pi * (1 - np.cos(scale))

class KCSRWorker(Process):
    """Worker process to compute K(s) - CSR(s) for N_samples samples."""

    def __init__(self, N_samples, conn):
        """N_samples is the number of samples of K(s) - CSR(s) to compute.
        conn is a multiprocessing.connection object for which .send() is
        available.
        """
        # Process.__init__() MUST be called before performing ANY OTHER action.
        super(KCSRWorker, self).__init__()
        self.N = N_samples
        self.conn = conn

    def run(self):
        """Computes N_samples samples of K(s) - CSR(s)

        After each sample is complete, the number of samples completed thus far
        is sent using the provided connection object (see __init__).
        Once all samples are complete, a dictionary of the form
        key = scale, value = K(scale) - CSR(scale) is sent.
        """
        stats = dict((s, []) for s in _test_scales)
        for i in range(self.N):
            dirs = isotropic_sample()
            Ks = k_function(_test_scales, dirs)
            CSRs = [CSR(s) for s in _test_scales]
            for (s, k, csr) in zip(_test_scales, Ks, CSRs):
                stats[s].append(k - csr)

            self.conn.send(i+1)

        self.conn.send(stats)
        self.conn.close()
        return


if __name__ == "__main__":
    desc = "Compute K(s) - CSR(s) statistics and provide upper/lower limits" + \
           " U and L such that P(x >= U) = P(x <= L) = significance"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-N", default=10000, type=int, dest="N_samples",
                        metavar="N",
                        help="Number of samples to generate " +
                              "(default: %(default)s)")
    parser.add_argument("-np, --processes", default=cpu_count(), type=int,
                        metavar="N",
                        dest="N_workers",
                        help="Number of processes to spawn " +
                             "(default: number of CPUs available)")

    args = parser.parse_args()
    N_workers = args.N_workers
    N_samples = args.N_samples

    stats = OrderedDict((s, []) for s in _test_scales)
    # Divide up as evenly as possible.
    N_per_worker = [int(round(N_samples / N_workers)), ] * N_workers
    # First worker gets slightly more or less work than the others if necessary.
    N_per_worker[0] -= sum(N_per_worker) - N_samples

    stdout.write("\rCompleted 0 of %d samples" % N_samples)
    stdout.flush()

    workers = []
    for i in range(N_workers):
        recv_conn, send_conn = Pipe(False)
        worker = KCSRWorker(N_per_worker[i], send_conn)
        worker.start()
        workers.append((worker, recv_conn))
        # NumPy's random number generator may seed based on the current system
        # time, or based on some other transient system data.
        sleep(2)

    # Almost all of the time is spent in this loop (but all the CPU usage is
    # offloaded to the worker processes - conn.recv() blocks with ~0% usage).
    N_completed = [0, ] * len(workers)
    while sum(N_completed) < N_samples:
        for i, (worker, conn) in enumerate(workers):
            if N_completed[i] < N_per_worker[i]:
                N_completed[i] = conn.recv()
                stdout.write("\rCompleted %d of %d samples"
                              % (sum(N_completed), N_samples))
                stdout.flush()
    print # Newline

    # The final item each worker sends is its computed k - CSR values.
    for (worker, conn) in workers:
        worker_stats = conn.recv()
        conn.close()
        for s in _test_scales:
            stats[s] += worker_stats[s]
        worker.terminate()
        worker.join()

    # The find_*_limit functions are not efficiently implemented, but take up
    # an entirely negligible fraction of the execution time.
    for s, scale_stats in stats.iteritems():
        U = find_upper_limit(scale_stats)
        L = find_lower_limit(scale_stats)

        pU = p_value(scale_stats, _UPPER(U))
        pL = p_value(scale_stats, _LOWER(L))

        print "%.4f:" % s
        print "U: %11.8f    P(x >= U) = %9.7f" % (U, pU)
        print "L: %11.8f    P(x <= L) = %9.7f" % (L, pL)
        print

    # for s, scale_stats in stats.iteritems():
    #     limits = np.linspace(np.min(scale_stats), np.max(scale_stats), num=1000)
        limits = np.linspace(2*U, 2*L, num=100)
        plt.clf()
        ps = [p_value(scale_stats, _UPPER(l)) for l in limits]
        plt.plot(limits, ps)
        #plt.ylim((0, 1.02))
        plt.savefig("P(x >= X) for scale %.4f.pdf" % s)

        plt.clf()
        ps = [p_value(scale_stats, _LOWER(l)) for l in limits]
        plt.plot(limits, ps)
        #plt.ylim((0, 1.02))
        plt.savefig("P(x <= X) for scale %.4f.pdf" % s)
