from glob import glob
from os.path import isfile
import re

import numpy as np

from . import stat_codes, _validate_code

# Rays saved as ox:<float>oy:<float> ... dz:<float>length:<float>.
_float_pattern=r"[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?"
_ray_pattern_str = r"^ox:(?P<ox>%(p)s)oy:(?P<oy>%(p)s)oz:(?P<oz>%(p)s)dx:(?P<dx>%(p)s)dy:(?P<dy>%(p)s)dz:(?P<dz>%(p)s)$" \
                   % {'p': _float_pattern}
_ray_pattern = re.compile(_ray_pattern_str)

def _read_rays(fname):
    dirs = []
    with open(fname) as f:
        for lnum, line in enumerate(f):
            match = re.match(_ray_pattern, line)
            if not match:
                raise RuntimeError("File contains invalid line\n  %s:%d"
                                   % (fname, lnum))
            dct = match.groupdict()
            d = np.array([dct['dx'], dct['dy'], dct['dz']],
                         dtype=np.float64)
            # Length is not specified in ray dumps.
            dirs.append(d)

    return np.vstack(dirs)

class _RaySamples(object):
    def __init__(self, ray_dir):
        super(_RaySamples, self).__init__()
        self._base_dir = ray_dir
        self._fnames = sorted(glob(self._base_dir + '/sample_*'))

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise TypeError("Only integer indexing allowed")

        return self._read_rays(i)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return len(self._fnames)

    def _read_rays(self, i):
        fname = self._fnames[i]
        return _read_rays(fname)

def _stat_exists(prefix, code, data_dir):
    _validate_code(code)
    return isfile(data_dir + '/' + prefix + code + '.csv')

def uniform_stat_exists(code, data_dir):
    return _stat_exists('reference_', code, data_dir)

def ray_stat_exists(code, data_dir):
    return _stat_exists('ray_', code, data_dir)

def bias_stat_exists(code, data_dir):
    return _stat_exists('biased_', code, data_dir)

def acceptable_error_exists(code, data_dir):
    return _stat_exists('error_', code, data_dir)

def results_exist(code, data_dir):
    if code == 'K':
        return _stat_exists('equivalence_', code, data_dir)
    else:
        return _stat_exists('noninferiority_', code, data_dir)

def ray_dumps_exist(ray_dir):
    rays = _RaySamples(ray_dir)
    return len(rays) > 0

def find_extant_data(data_dir):
    """
    Return a dict of dicts of flags for the existence of test data.

    Valid keys for the top-level dict refer to the source distribution
      uniform - refers to the uniform (reference) distribution
      biased - refers to the biased test distribution
      rays - refers to the GRACE ray data
      error - the maximum acceptable error, computed from the biased and
              reference distributions.

    For each sub-dict, any of the statistic codes 'W', 'An' or 'Gn' may be used
    to query whether data for that statistic exists for the given distribution.
    The statistic code 'K' is valid only for 'uniform' and 'rays'
    """
    extant = dict()
    extant['uniform'] = dict((c, False) for c in stat_codes)
    extant['ray'] = dict((c, False) for c in stat_codes)
    extant['biased'] = dict((c, False) for c in stat_codes if c != 'K')
    extant['error'] = dict((c, False) for c in stat_codes if c != 'K')
    for code in stat_codes:
        if uniform_stat_exists(code, data_dir):
            extant['uniform'][code] = True
        if ray_stat_exists(code, data_dir):
            extant['ray'][code] = True

        if code != 'K':
            if bias_stat_exists(code, data_dir):
                extant['biased'][code] = True
            if acceptable_error_exists(code, data_dir):
                extant['error'][code] = True

    return extant

def write_acceptable_error(code, e, bf, data_dir):
    """
    Write the provided maximum acceptable error.

    e should be a positive scalar value specifying the maximum acceptable
    fractional deviation from a Mann-Whitney statistic of 0.5.
    bf is the bias-fraction for the dataset from which e was derived.

    The Ripley K-statistic is not valid here.
    """
    if code == 'K':
        raise ValueError("K-function does not pre-compute acceptable error")
    if e <= 0 or bf < 0 or e >= 1.0 or bf >= 0.5:
        raise ValueError("e must be in (0, 1); f must be in [0, 0.5)")

    fname = data_dir + '/error_' + code + '.csv'
    with open(fname, 'w') as f:
        f.write('# Specified bias fraction\n')
        f.write('%.4f\n' % (bf, ))
        f.write('# e\n')
        f.write('%.6f\n' % (e, ))

def read_acceptable_error(code, data_dir):
    """
    Return the maximum acceptable error for the given statistic.

    Return two values; the maximum acceptable error, and the bias fraction to
    which it corresponds.

    The Ripley K-statistic is not valid here.
    """
    if code == 'K':
        raise ValueError("K-function has no pre-computed acceptable error")

    fname = data_dir + '/error_' + code + '.csv'
    f, e = np.loadtxt(fname, delimiter=', ')
    return e, f

def write_ray_stats(code, data, data_dir):
    """
    Write the provided GRACE ray statistics.

    For Ripley-K statistics (code 'K'), data should be a 2D array where each
    row is a different sample, and each column a different scale size.

    Otherwise, data should be a 1D array, where each element gives the
    statistic for a different sample.
    """
    if code != 'K' and data.ndim > 2:
        raise ValueError("Only K-function data may be multidimensional")

    fname = data_dir + '/ray_' + code + '.csv'
    return np.savetxt(fname, data, delimiter=', ')

def read_ray_stats(code, data_dir):
    """
    Return the GRACE ray statistics for the given statistic.

    If code is 'K', return a 2D array; each row gives the K-statistic for a
    different sample, and each column corresponds to each scale.

    Otherwise, return a 1D array; each element gives the statistic for a
    different sample.
    """
    fname = data_dir + '/ray_' + code + '.csv'
    return np.loadtxt(fname, delimiter=', ')

def write_reference_stats(code, data, data_dir):
    """
    Write the provided reference statistics.

    For Ripley-K statistics (code 'K'), data should be a 2D array where each
    row is a different sample, and each column a different scale size.

    Otherwise, data should be a 1D array, where each element gives the
    statistic for a different sample.
    """
    if code != 'K' and data.ndim > 2:
        raise ValueError("Only K-function data may be multidimensional")

    fname = data_dir + '/reference_' + code + '.csv'
    return np.savetxt(fname, data, delimiter=', ')

def read_reference_stats(code, data_dir):
    """
    Return the reference statistics for the given statistic.

    If code is 'K', return a 2D array; each row gives the K-statistic for a
    different sample, and each column corresponds to each scale.

    Otherwise, return a 1D array; each element gives the statistic for a
    different sample.
    """
    fname = data_dir + '/reference_' + code + '.csv'
    return np.loadtxt(fname, delimiter=', ')

def write_biased_stats(code, data, data_dir):
    """
    Write the provided biased statistics.

    Data should be a 2D array, where each row corresponds to a different number
    of bias modes, and each element within a row to a different sample.

    The Ripley K-statistic is not valid here.
    """
    if code == 'K':
        raise ValueError("No appropriate biased samples for K-function")
    if data.ndim > 2:
        raise ValueError("data must be a two-dimensional array")

    fname = data_dir + '/biased_' + code + '.csv'
    with open(fname, 'w') as f:
        for (m, sample_stats) in enumerate(data):
            f.write('# %d bias mode(s)\n' % (m + 1))
            f.write(', '.join(str(x) for x in sample_stats) + '\n')

def read_biased_stats(code, data_dir):
    """
    Return the biased statistics for the given statistic.

    Return a 2D array; each row contains statistics for a different number
    of modes, and each element within a row is the statistic for a different
    sample.

    The Ripley K-statistic is not valid here.
    """
    if code == 'K':
        raise ValueError("No appropriate biased samples for K-function")

    fname = data_dir + '/biased_' + code + '.csv'
    return np.loadtxt(fname, delimiter=', ')
