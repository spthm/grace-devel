from __future__ import print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count
from os.path import isdir, isfile
from sys import argv

import analysis
import analysis.io as io
import analysis.samples as smpl
import analysis.test as t

stat_codes = analysis.stat_codes

def print_if_missing(extant_data, params):
    missing = dict()

    stats = [c for (c, exists) in extant_data['uniform'].items() if not exists]
    if stats:
        missing['uniform reference'] = stats

    stats = [c for (c, exists) in extant_data['biased'].items()
             if not exists and c != 'K']
    if stats:
        missing['bias reference'] = stats

    stats = [c for (c, exists) in extant_data['ray'].items() if not exists]
    if stats:
        missing['ray'] = stats

    stats = [c for (c, exists) in extant_data['error'].items() if not exists]
    if stats:
        missing['maximum acceptable error'] = stats

    if missing:
        print("Will (re)compute statistics for:")
        for k, stats in missing.items():
            if 'K' in stats:
                i = stats.index('K')
                stats[i] = "K(%d)" % params.n_scales
            print("  " + k + ": " + ", ".join(stats))
        print("With default arguments, this may take several hours")

    return True if missing else False

def set_forced(extant_data, params):
    if params.force_uniform or params.force_all:
        for k in extant_data['uniform']:
            extant_data['uniform'][k] = False
    if params.force_biased or params.force_all:
        for k in extant_data['biased']:
            extant_data['biased'][k] = False
    if params.force_rays or params.force_all:
        for k in extant_data['ray']:
            extant_data['ray'][k] = False

    # This may have changed which maximum acceptable error values need updating.
    # They depend on both the biased and uniform reference data. Loop over codes
    # in biased, since 'K' is not applicable for 'biased' or 'error'.
    for c in extant_data['biased']:
        if not (extant_data['biased'][c] and extant_data['uniform'][c]):
            extant_data['error'][c] = False

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-fu", "--force-uniform",
                    help="Recompute all uniform random reference statistics",
                    action="store_true",
                    dest="force_uniform")
parser.add_argument("-fb", "--force-biased",
                    help="Recompute all biased reference statistics",
                    action="store_true",
                    dest="force_biased")
parser.add_argument("-fr", "--force-rays",
                    help="Recompute all GRACE ray statistics",
                    action="store_true",
                    dest="force_rays")
parser.add_argument("-fa", "--force-all",
                    help="Recompute all statistics (uniform, biased and rays)",
                    action="store_true",
                    dest="force_all")
parser.add_argument("-np", "--num_procs",
                    help=("The number of processes to spawn when computing "
                          "statistics"),
                    type=int,
                    default=cpu_count(),
                    metavar='N',
                    dest="n_procs")
parser.add_argument("-nk", "--num_scales",
                    help=("The number of evenly-spaced scales in (0, pi) at "
                          "which to test Ripley's K-function"),
                    type=int,
                    default=7,
                    metavar='N',
                    dest="n_scales")
parser.add_argument("-c", "--confidence",
                    help="The confidence level of all statistical tests",
                    type=float,
                    default=0.995)
parser.add_argument("-nm", "--num_modes",
                    help=("The maximum number of preferred directions to "
                          "choose when computing biased reference statistics"),
                    type=int,
                    default=4,
                    metavar='N',
                    dest="max_n_modes")
parser.add_argument("-f", "--bias_fraction",
                    help=("The fractional number of directions which are "
                          "biased when generating biased reference "
                          "statistics"),
                    type=float,
                    default=0.005,
                    metavar='FRACTION')
parser.add_argument("-ke", "--k_error",
                    help=("The maximum acceptable fractional deviation from "
                          "the complete spatial randomness value of Ripley's "
                          "K-statistic"),
                    type=float,
                    default=0.001,
                    metavar='FRACTION')
parser.add_argument("-ns", "--num_samples",
                    help=("The number of samples if computing uniform or "
                          "biased statistics"),
                    type=int,
                    default=2000,
                    metavar='N',
                    dest="n_samples")
parser.add_argument("-ss", "--sample_size",
                    help=("The number of directions in each sample if "
                          "computing uniform or biased statistics. "
                          "It must match the number of rays in each GRACE "
                          "ray dump"),
                    type=int,
                    default=9600,
                    metavar='N')
parser.add_argument("-dd", "--data-dir",
                    help=("The directory data should be read from and written "
                          "to"),
                    type=str,
                    default='./csv',
                    metavar='DIR',
                    dest='data_dir')
parser.add_argument("-pd", "--plots-dir",
                    help="The directory in which plots should be saved",
                    type=str,
                    default='./plots',
                    metavar='DIR',
                    dest='plots_dir')
parser.add_argument("-rd", "--ray-dir",
                    help=("The directory from which GRACE ray dumps should be "
                          "read, if necessary"),
                    type=str,
                    default='./ray-dumps',
                    metavar='DIR',
                    dest='ray_dir')
params = parser.parse_args()

quit = False
if not isdir(params.data_dir):
    print("Specified data directory does not exist:\n  %s" % params.data_dir)
    quit = True
if not isdir(params.plots_dir):
    print("Specified plots directory does not exist:\n  %s" % params.plots_dir)
    quit = True

extant_data = io.find_extant_data(params.data_dir)
set_forced(extant_data, params)
if not all(extant_data['ray'].values()):
    if not io.ray_dumps_exist(params.ray_dir):
        print("Do not have all ray statistics")
        print("No ray dumps in %s" % params.ray_dir)
        print("Please compile and run ray_dump.cu, then re-run this script")
        quit = True

if quit:
    exit()

# TODO: check extant data and make sure ref/bias/ray Ks use same number of
# scales, if present. If not, flag them _all_ as false in extant_data.

any_missing = print_if_missing(extant_data, params)

if any_missing:
    # Do the rays first, as it may fail if the number of rays in each sample
    # does not match params.sample_size.
    codes = [c for (c, exists) in extant_data['ray'].items() if not exists]
    smpl.generate_ray_statistics(codes, params)

    codes = [c for (c, exists) in extant_data['uniform'].items() if not exists]
    smpl.generate_uniform_statistics(codes, params)

    codes = [c for (c, exists) in extant_data['biased'].items() if not exists]
    smpl.generate_biased_statistics(codes, params)

    codes = [c for (c, exists) in extant_data['error'].items() if not exists]
    smpl.generate_acceptable_errors(codes, params)

t.plot_k_results(params.confidence, params.k_error, params.data_dir,
                 params.plots_dir)
t.print_W_result(params.confidence, params.data_dir)
t.print_An_result(params.confidence, params.data_dir)
t.print_Gn_result(params.confidence, params.data_dir)
