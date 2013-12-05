from os.path import isfile
from itertools import count

import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'times'
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'medium'
import matplotlib.pyplot as plt

from profileparser import ProfileResult

#base_string = "profile_treeclimb_{0}.log"
base_string = "profile_tree_{0}.log"
profile_results = []
# Read in as many files as exist.
for i in count(1):
    fname = base_string.format(str(i))
    try:
        with open(fname) as f:
            print "Reading in data for " + fname
            result = ProfileResult(f)
            result.generate_stats()
            profile_results.append(result)
    except IOError:
        if isfile(fname):
            raise IOError("File " + fname + " exists but cannot be opened.")
        else:
            # Assume we've read in all the files.
            break

for (i,result) in enumerate(profile_results):
    fig1 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    ax4 = fig4.add_subplot(111)

    fig1.suptitle("Fraction of time taken per kernel for random points")
    fig3.suptitle("Log-Log cummulative time for random points")
    fig4.suptitle("Log-Log kernel time")

    first_ID = result.kernel_speed_order[0]
    balanced_ID = result.kernel_ID("balanced tree AABBs")
    total_ID = result.kernel_ID("total (inc. memory ops)")

    # Set cummulative time > 0 so log works.
    cummulative_time = 1E-80*np.ones_like(result.timings[0])
    cummulative_frac = np.zeros_like(result.timings[0])
    for ID in result.kernel_speed_order:
        label = result.kernel_name(ID)[0].upper() + result.kernel_name(ID)[1:]
        # Balanced run distinct from build time.
        if ID != balanced_ID:
            # For non-total result we need to plot the cummulative result.
            if ID != total_ID:
                line, = ax1.plot(result.levels-1,
                                 cummulative_frac +
                                 result.timings[ID]/result.timings[total_ID],
                                 label=label,
                                 linewidth=0.5)
                ax1.fill_between(result.levels-1,
                                 cummulative_frac,
                                 cummulative_frac +
                                 result.timings[ID]/result.timings[total_ID],
                                 facecolor=line.get_color(),
                                 alpha=0.5)
                cummulative_frac += result.timings[ID]/result.timings[total_ID]
                line, ax3.plot(result.levels-1,
                         np.log10(result.timings[ID]+cummulative_time),
                         label=label,
                         linewidth=0.5)
                ax3.fill_between(result.levels-1,
                                 np.log10(cummulative_time),
                                 np.log10(result.timings[ID]+cummulative_time),
                                 facecolor=line.get_color(),
                                 alpha=0.5)
                if ID == first_ID:
                    cummulative_time = np.zeros_like(result.timings[ID])
                cummulative_time += result.timings[ID]
            # For the total, the result is implicitly cummulative.
            else:
                line, = ax1.plot(result.levels-1,
                                 cummulative_frac +
                                 (result.timings[total_ID] - result.timings.sum(axis=0) + result.timings[balanced_ID])/result.timings[total_ID],
                                 label="Memory ops",
                                 linewidth=0.5)
                ax1.fill_between(result.levels-1,
                                 cummulative_frac,
                                 cummulative_frac +
                                 (result.timings[total_ID] - cummulative_frac)/result.timings[total_ID],
                                 facecolor=line.get_color(),
                                 alpha=0.5)
                cummulative_frac += (result.timings[total_ID] - cummulative_frac) / result.timings[total_ID]

                line, ax3.plot(result.levels-1,
                               np.log10(result.timings[ID]),
                               label=label,
                               linewidth=0.5)
                ax3.fill_between(result.levels-1,
                                 np.log10(cummulative_time),
                                 np.log10(result.timings[ID]),
                                 facecolor=line.get_color(),
                                 alpha=0.5)
            # All non-cummulative times plotted identically.
            ax4.plot(result.levels-1,
                     np.log10(result.timings[ID]),
                     label=label)
    # Add balanced-tree AABB times to non-cummulative plot.
    # Happens last to maintain colours between plots.
    label = (result.kernel_name(balanced_ID)[0].upper() +
             result.kernel_name(balanced_ID)[1:])
    ax4.plot(result.levels-1,
             np.log10(result.timings[balanced_ID]),
             label=label)

    ax1.set_xlabel(r"$\log_{\,2} (N_{\mathrm{leaves}})$")
    ax1.set_ylabel(r"$\bar{t}_\mathrm{kernel} \; / \; \Sigma_i \;  \bar{t}_i$")
    ax1.set_xlim(min(result.levels-1), max(result.levels-1))
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", frameon=False)

    ax3.set_xlabel(r"$\log_{\,2} (N_{\mathrm{leaves}})$")
    ax3.set_ylabel(r"$\log_{\,10}" +
                   r"(\Sigma \;  \bar{t}_\mathrm{kernel} / 1 \mathrm{ms})$")
    ax3.set_xlim(min(result.levels-1), max(result.levels-1))
    ax3.set_ylim(np.log10(min(result.timings[first_ID])),
                 np.log10(max(result.timings[total_ID])))
    ax3.legend(loc="upper left", frameon=False)

    ax4.set_xlabel(r"$\log_{\,2} N_{\mathrm{leaves}}$")
    ax4.set_ylabel(r"$\log_{\,10}(\bar{t}_\mathrm{kernel} / 1 \mathrm{ms})$")
    ax4.set_xlim(min(result.levels-1), max(result.levels-1))
    ax4.set_ylim(np.log10(np.amin(result.timings)),
                 np.log10(np.amax(result.timings)))
    ax4.legend(loc="upper left", frameon=False)

    fig1.savefig("fractional.pdf")
    fig3.savefig("log-log cummulative.pdf")
    fig4.savefig("log-log individual.pdf")

plt.show()
