from __future__ import print_function

from collections import Sequence
from multiprocessing import Pool
import signal

import numpy as np

from . import _validate_code
from .biased import symmetric_biased_directions, nonsymmetric_biased_directions
from .hypothesis import _mann_whitney
from .io import _RaySamples
from .io import read_biased_stats, read_reference_stats
from .io import write_ray_stats, write_biased_stats, write_reference_stats
from .io import write_acceptable_error
from .random import n_random_directions
from .statistics import k_scales, k_function, rayleigh, An_Gn
from .util import print_overwrite

class _Callable(object):
    """A simple function object."""
    def __init__(self, function, pre_args=None, post_args=None, ret_slice=None):
        """
        Provide the function to be called and, optionally, leading and trailing
        arguments. The functions return values may also, optionally, be slices.

        function must be a callable.
        ret_slice must be a slice or int, if provided.

        On calling an instance of _Callable as callable(x), it will return
            function([[*]pre_args,] [*]x[, [*]post_args])[ret_slice]
        where [*] denotes that argument expansion only occurs if a sequence
        is provided. Further, if pre_args or post_args are not provided, no
        argument is passed to function in their place. Finally, if ret_slice
        is not provided, not slicing occurs.

        To provide a sequence, seq, as a single post argument, for example, one
        must initialize with post_args = [seq, ]. Using instead post_args=seq
        would result in the argument-expansion of seq when passed to function.
        """
        super(_Callable, self).__init__()

        if not callable(function):
            raise TypeError("function must be callable")
        if ret_slice is not None:
            if not (isinstance(ret_slice, slice) or isinstance(ret_slice, int)):
               raise TypeError("ret_slice must be slice or int if provided")

        self.f = function
        self.slc = ret_slice

        if pre_args is None:
            self.pre = None
        else:
            self.pre = self._make_sequence(pre_args)

        if post_args is None:
            self.post = None
        else:
            self.post = self._make_sequence(post_args)

    def __call__(self, x):
        args = []

        # Falsey objects are valid arguments.
        if self.pre is not None:
            args += self.pre

        args += self._make_sequence(x)


        # Falsey objects are valid arguments.
        if self.post is not None:
            args += self.post

        # self.slc == 0 is a valid slice!
        if self.slc is not None:
            return self.f(*args)[self.slc]
        else:
            return self.f(*args)

    def _make_sequence(self, x):
        if not isinstance(x, Sequence) or isinstance(x, str):
            return [x, ]
        return x

def _fetch_ray_sample(i, ray_samples, sample_size):
    """
    Return the ith sample.

    Raise ValueError if the sample size is not equal to sample_size.
    """
    sample = ray_samples[i]
    if len(sample) != sample_size:
        raise RuntimeError("%dth ray sample size is %d, but %d specified"
                           % (i, len(sample), sample_size))
    return sample

def _biased_sample_fn(code):
    if code in ('W', 'An'):
        # return lambda s, m, f: nonsymmetric_biased_directions(s, m, f)[0]
        return _Callable(nonsymmetric_biased_directions, ret_slice=0)
    else: # code == 'Gn'
        # return lambda s, m, f: symmetric_biased_directions(s, m, f)[0]
        return _Callable(symmetric_biased_directions, ret_slice=0)

def _stat_fn(code):
    if code == 'W':
        # return lambda s: rayleigh(s)
        return _Callable(rayleigh)
    elif code == 'An':
        # return lambda s: An_Gn(sample)[0]
        return _Callable(An_Gn, ret_slice=0)
    else: # code == 'Gn'
        # return lambda s: An_Gn(sample)[1]
        return _Callable(An_Gn, ret_slice=1)

def _fractional_mann_whitney_diff(bias, refs, nprocs):
    if bias.ndim > 1:
        # Just take the 1-mode bias result. It always has the farthest-from
        # uniform result.
        bias = bias[0]
    wxy, _ = _mann_whitney(bias, refs, nprocs=nprocs)
    print("Wxy: %.4f" % wxy)
    return abs(wxy - 0.5) / 0.5

def _generate_acceptable_error(code, params):
    _validate_code(code)
    if code == 'K':
        raise ValueError("K not applicable for precomputed error")

    refs = read_reference_stats(code, params.data_dir)
    bias = read_biased_stats(code, params.data_dir)

    print_overwrite("Computing maximum error in " + code + " from bias...")
    e = _fractional_mann_whitney_diff(bias, refs, params.n_procs)
    write_acceptable_error(code, e, params.bias_fraction, params.data_dir)

def _ctrlc_exit(pool):
    """
    Terminate all workers in the pool and exit with code 1.
    """
    print()
    print("SIGINT detected. Terminating worker processes...")
    pool.terminate()
    pool.join()
    exit(1)

def _worker(args):
    """
    Return statistic for (stat_func, sample_func, sample_func_args) in args.
    """
    # Worker processes should ignore SIGINT (e.g. CTRL+C).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The seed is set on import, i.e. before we fork processes.
    np.random.seed()
    stat_fn, sample_fn, sample_args = args
    sample = sample_fn(sample_args)
    stat = stat_fn(sample)
    return stat

def _generate_ray_k_statistics(params):
    rays = _RaySamples(params.ray_dir)
    n_samples = len(rays)
    sample_size = params.sample_size
    scales = k_scales(params.n_scales)

    Ks = np.empty((n_samples, len(scales)), dtype=np.float64)
    # stat_fn = lambda s: k_function(scales, s)
    stat_fn = _Callable(k_function, pre_args=[scales, ])
    # sample_fn = lambda i: _sample_size_check(rays[i], sample_size)
    sample_fn = _Callable(_fetch_ray_sample, post_args=[rays, sample_size])

    pool = Pool(processes=params.n_procs)
    args = ((stat_fn, sample_fn, i) for i in range(n_samples))
    try:
        for i, sample_K in enumerate(pool.imap(_worker, args)):
            Ks[i] = sample_K
            print_overwrite("Computed K-statistic for ray sample %d of %d"
                            % (i+1, n_samples))
    except KeyboardInterrupt:
        _ctrlc_exit(pool)
    print()

    write_ray_stats('K', Ks, params.data_dir)

def _generate_uniform_k_statistics(params):
    n_samples = params.n_samples
    sample_size = params.sample_size
    scales = k_scales(params.n_scales)

    Ks = np.empty((n_samples, len(scales)), dtype=np.float64)
    # stat_fn = lambda s: k_function(scales, s)
    stat_fn = _Callable(k_function, pre_args=[scales, ])
    sample_fn = n_random_directions

    pool = Pool(processes=params.n_procs)
    args = ((stat_fn, sample_fn, sample_size) for i in range(n_samples))
    try:
        for i, sample_K in enumerate(pool.imap(_worker, args)):
            Ks[i] = sample_K
            print_overwrite("Computed K-statistic for uniform sample %d of %d"
                            % (i+1, n_samples))
    except KeyboardInterrupt:
        _ctrlc_exit(pool)
    print()

    write_reference_stats('K', Ks, params.data_dir)

def _generate_biased_statistics(code, params):
    _validate_code(code)
    if code == 'K':
        raise ValueError("K not applicable for biased statistics")

    n_samples = params.n_samples
    sample_size = params.sample_size
    max_n_modes = params.max_n_modes
    bias_f = params.bias_fraction

    biased = np.empty((max_n_modes, n_samples), dtype=np.float64)
    m_biased = np.empty(n_samples, dtype=np.float64)
    stat_fn = _stat_fn(code)
    sample_fn = _biased_sample_fn(code)

    pool = Pool(processes=params.n_procs)
    for m in range(1, max_n_modes + 1):
        sample_args = (sample_size, m, bias_f)
        args = ((stat_fn, sample_fn, sample_args) for i in range(n_samples))
        try:
            for i, stat in enumerate(pool.imap(_worker, args)):
                m_biased[i] = stat
                print_overwrite("Computed " + code + "-statistic for biased "
                                "sample %d of %d, mode %d of %d"
                                % (i+1, n_samples, m, max_n_modes))
        except KeyboardInterrupt:
            _ctrlc_exit(pool)
        biased[m-1,:] = m_biased
    print()

    write_biased_stats(code, biased, params.data_dir)

def _generate_ray_statistics(code, params):
    _validate_code(code)
    rays = _RaySamples(params.ray_dir)
    n_samples = len(rays)
    sample_size = params.sample_size

    stats = np.empty(n_samples, dtype=np.float64)
    stat_fn = _stat_fn(code)
    # sample_fn = lambda i: _sample_size_check(rays[i], sample_size)
    sample_fn = _Callable(_fetch_ray_sample, post_args=[rays, sample_size])

    pool = Pool(processes=params.n_procs)
    args = ((stat_fn, sample_fn, i) for i in range(n_samples))
    try:
        for i, stat in enumerate(pool.imap(_worker, args)):
            stats[i] = stat
            print_overwrite("Computed " + code + "-statistic for ray sample "
                            "%d of %d" % (i+1, n_samples))
    except KeyboardInterrupt:
        _ctrlc_exit(pool)
    print()

    write_ray_stats(code, stats, params.data_dir)

def _generate_uniform_statistics(code, params):
    _validate_code(code)
    n_samples = params.n_samples
    sample_size = params.sample_size

    stats = np.empty(n_samples, dtype=np.float64)
    stat_fn = _stat_fn(code)
    sample_fn = n_random_directions

    pool = Pool(processes=params.n_procs)
    args = ((stat_fn, sample_fn, sample_size) for i in range(n_samples))
    try:
        for i, stat in enumerate(pool.imap(_worker, args)):
            stats[i] = stat
            print_overwrite("Computed " + code + "-statistic for uniform "
                            "sample %d of %d" % (i+1, n_samples))
    except KeyboardInterrupt:
        _ctrlc_exit(pool)
    print()

    write_reference_stats(code, stats, params.data_dir)

def generate_acceptable_errors(codes, params):
    # Do not bother with K. It is trivial to compute when plotting.
    if 'W' in codes:
        _generate_acceptable_error('W', params)
    if 'An' in codes:
        _generate_acceptable_error('An', params)
    if 'Gn' in codes:
        _generate_acceptable_error('Gn', params)

def generate_biased_statistics(codes, params):
    # Do not do K. We don't have a simple method of generating biased rays
    # for that statistic.
    if 'W' in codes:
        _generate_biased_statistics('W', params)
    if 'An' in codes:
        _generate_biased_statistics('An', params)
    if 'Gn' in codes:
        _generate_biased_statistics('Gn', params)

def generate_ray_statistics(codes, params):
    if 'K' in codes:
        _generate_ray_k_statistics(params)
    if 'W' in codes:
        _generate_ray_statistics('W', params)
    if 'An' in codes:
        _generate_ray_statistics('An', params)
    if 'Gn' in codes:
        _generate_ray_statistics('Gn', params)

def generate_uniform_statistics(codes, params):
    if 'K' in codes:
        _generate_uniform_k_statistics(params)
    if 'W' in codes:
        _generate_uniform_statistics('W', params)
    if 'An' in codes:
        _generate_uniform_statistics('An', params)
    if 'Gn' in codes:
        _generate_uniform_statistics('Gn', params)
