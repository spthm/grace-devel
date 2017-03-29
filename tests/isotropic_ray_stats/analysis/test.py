from __future__ import print_function

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'lmr'
mpl.rcParams['font.sans-serif'] = 'lmss'
mpl.rcParams['font.size'] = 10.0
mpl.rcParams['xtick.labelsize'] = 8.0
mpl.rcParams['ytick.labelsize'] = 8.0
mpl.rcParams['savefig.dpi'] = 400

import matplotlib.pyplot as plt
import numpy as np

from .hypothesis import equivalence_test, nonnormal_noninferiority_test
from .io import read_acceptable_error, read_ray_stats, read_reference_stats
from .statistics import k_scales, k_null
from .util import print_overwrite, print_overwrite_done

def _interval_to_error_bars(low, high):
    y = (low + high) / 2.
    low_err = y - low
    high_err = high - y
    return y, [low_err, high_err]

def _read_ray_ref_stats(code, data_dir):
    ray = read_ray_stats(code, data_dir)
    ref = read_reference_stats(code, data_dir)
    return ray, ref

def _noninferiority_result(ray_data, ref_data, confidence, e):
    # For An, Gn and W, larger values are worse.
    r = nonnormal_noninferiority_test(ray_data, ref_data, e, cl=confidence,
                                      inferior='larger')
    noninf, _, _, test, crit = r
    return noninf, test, crit

def _print_noninferiority_result(reject, test_stat, crit, confidence,
                                 description):
    # The <= and > signs here are only correct for a nonnormal noninferiority
    # test, or a normal noninferiority test where inferior == smaller.
    if reject:
        print("Ray " + description + " distribution noninferior to reference "
              "at %.4f%% confidence level:" % (100 * confidence, ))
        print("  %.4f > %.4f" % (test_stat, crit))
    else:
        print("No evidence to reject null hypothesis of ray " + description +
              " inferior to\nreference at %.3f%% confidence level:"
              % (100 * confidence, ))
        print("  %.4f <= %.4f" % (test_stat, crit))
    print()

def plot_k_results(confidence, e, data_dir, plot_dir):
    """
    Compute and save plots for the K-statistic results.

    confidence - the confidence level of the test, in (0, 1).
    e - the largest acceptable underdensity and overdensity at a given scale
        which is still considered equivalent, expressed as a fractional value of
        the null (complete spatial randomness) result.
    """
    ray_ks, ref_ks = _read_ray_ref_stats('K', data_dir)

    if ref_ks.shape[1] != ray_ks.shape[1]:
        msg = ("Number of scales in ray and reference do not match (%d, %d)"
                % (ray_ks.shape[1], ref_ks.shape[1]))
        raise RuntimeError(msg)

    n_scales = ref_ks.shape[1]
    scales = k_scales(n_scales)

    # Magnitude of region of indifference in each direction.
    reject_low = e * k_null(scales)
    reject_high = e * k_null(scales)

    clims_low = np.empty(n_scales, dtype=np.float64)
    clims_high = np.empty(n_scales, dtype=np.float64)
    equivs = np.empty(n_scales, dtype=np.bool)
    for i, s in enumerate(scales):
        equiv, Clow, Chigh = equivalence_test(ray_ks[:,i], ref_ks[:,i],
                                              reject_low[i], reject_high[i],
                                              cl=confidence)
        equivs[i] = equiv
        clims_low[i] = Clow
        clims_high[i] = Chigh
        print_overwrite("Computed confidence interval for scale %d of %d"
                         % (i+1, len(scales)))
    print_overwrite_done()

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y, yerr = _interval_to_error_bars(clims_low, clims_high)
    ax.errorbar(scales, y, yerr=yerr, ecolor='k', fmt='none')
    ax.axhline(y=0, xmin=0, xmax=np.pi, c='k', linewidth=0.5, linestyle=':')
    fine_scales = np.linspace(0.0, np.pi, 100)
    fill = e * k_null(fine_scales)
    ax.fill_between(fine_scales, -fill, fill, interpolate=True,
                    edgecolor='none',facecolor='grey', alpha=0.4)

    ax.set_ylim(2. * min(clims_low), 2. * max(clims_high))
    ax.set_ylabel(r"$\hat{K} - \hat{K}_\mathrm{csr}$")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax.set_xlim(0.0, np.pi)
    ax.set_xticks(np.linspace(0.0, np.pi, 7))
    ax.set_xticklabels(['0', r'$\frac{\pi}{6}$', r'$\frac{2\pi}{6}$',
                        r'$\frac{3\pi}{6}$', r'$\frac{4\pi}{6}$',
                        r'$\frac{5\pi}{6}$', r'$\pi$'])
    ax.set_xlabel(r"$s$")

    plt.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.92)
    fig.savefig(plot_dir + "/ks_confidence.pdf")

def _print_result(code, confidence, data_dir):
    rays, refs = _read_ray_ref_stats(code, data_dir)
    e, _ = read_acceptable_error(code, data_dir)

    desc = code if code != 'W' else 'Rayleigh'

    print_overwrite("Performing noninferiority test for " + desc +
                    " statistic...")
    noninf_W, W_test, W_crit = _noninferiority_result(rays, refs, confidence, e)
    print_overwrite_done()

    _print_noninferiority_result(noninf_W, W_test, W_crit, confidence,
                                 desc + "-statistic")

def print_W_result(confidence, data_dir):
    """
    Compute and print to console the W-statistic result.

    confidence - the confidence level of the test, in (0, 1).
    """
    _print_result('W', confidence, data_dir)
    # ray_Ws, ref_Ws = _read_ray_ref_stats('W', data_dir)
    # e, _ = read_acceptable_error('W', data_dir)

    # print_overwrite("Performing noninferiority test for Rayleigh statistic...")
    # noninf_W, W_test, W_crit = _noninferiority_result(ray_Ws, ref_Ws,
    #                                                   confidence, e)
    # print_overwrite_done()

    # _print_noninferiority_result(noninf_W, W_test, W_crit, confidence,
    #                              "Rayleigh-statistic")

def print_An_result(confidence, data_dir):
    """
    Compute and print to console the An-statistic result.

    confidence - the confidence level of the test, in (0, 1).
    """
    _print_result('An', confidence, data_dir)
    # ray_Ans, ref_Ans = _read_ray_ref_stats('An', data_dir)
    # e, _ = read_acceptable_error('An', data_dir)

    # print_overwrite("Performing noninferiority test for An statistic...")
    # noninf_An, An_test, An_crit = _noninferiority_result(ray_Ans, ref_Ans,
    #                                                      confidence, e)
    # print_overwrite_done()

    # _print_noninferiority_result(noninf_An, An_test, An_crit, confidence,
    #                              "An-statistic")

def print_Gn_result(confidence, data_dir):
    """
    Compute and print to console the Gn-statistic result.

    confidence - the confidence level of the test, in (0, 1).
    """
    _print_result('Gn', confidence, data_dir)
    # ray_Gns, ref_Gns = _read_ray_ref_stats('Gn', data_dir)
    # e, _ = read_acceptable_error('Gn', data_dir)

    # print_overwrite("Performing noninferiority test for Gn statistic...")
    # noninf_Gn, Gn_test, Gn_crit = _noninferiority_result(ray_Gns, ref_Gns,
    #                                                      confidence, e)
    # print_overwrite_done()

    # _print_noninferiority_result(noninf_Gn, Gn_test, Gn_crit, confidence,
    #                              "Gn-statistic")
