from os.path import isfile
from itertools import count

import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from profileparser import ProfileResult

#base_string = "profile_treeclimb_{0}.log"
base_string = "profile_tree_{0}.log"
profile_results = []
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
            break

for (i,result) in enumerate(profile_results):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    first_ID = result.kernel_speed_order[0]
    balanced_ID = result.kernel_ID("balanced tree AABBs")
    total_ID = result.kernel_ID("loop (inc. memory ops)")

    cummulative_time = np.zeros_like(result.timings[0])
    for ID in result.kernel_speed_order:
        label = result.kernel_name(ID)[0].upper() + result.kernel_name(ID)[1:]
        if ID != balanced_ID:
            # Balanced run distinct from build time.
            if ID != total_ID:
                # Need to plot the cummulative result.
                line1, = ax1.plot(result.levels,
                                  result.timings[ID]+cummulative_time,
                                  label=label,
                                  linewidth=0.5)
                ax1.fill_between(result.levels,
                                 cummulative_time,
                                 result.timings[ID]+cummulative_time,
                                 facecolor=line1.get_color(),
                                 alpha=0.5)
                cummulative_time += result.timings[ID]
            else:
                # Result is implicitly cummulative.
                line1, = ax1.plot(result.levels,
                                  result.timings[ID],
                                  label=label,
                                  linewidth=0.5)
                ax1.fill_between(result.levels,
                                 cummulative_time,
                                 result.timings[ID],
                                 facecolor=line1.get_color(),
                                 alpha=0.5)
            # Non-cummulative times plotted identically.
            ax2.plot(result.levels,
                     np.log10(result.timings[ID]),
                     label=label)
    # Add balanced-tree AABB times to non-cummulative plot.
    # Happens last to maintain colours between plots.
    label = (result.kernel_name(balanced_ID)[0].upper() +
             result.kernel_name(balanced_ID)[1:])
    ax2.plot(result.levels,
             np.log10(result.timings[balanced_ID]),
             label=label)

    ax1.set_xlabel(r"$\log_{\,2} (N_{\mathrm{leaves}}) + 1$")
    ax1.set_ylabel(r"$\Sigma \; \bar{t}_\mathrm{kernel} \; \mathrm{[ms]}$")
    ax1.set_xlim(min(result.levels), max(result.levels))
    ax1.set_ylim(min(result.timings[first_ID]),
                 max(result.timings[total_ID]))
    ax1.legend(loc="upper left")

    ax2.set_xlabel(r"$\log_{\,2} (N_{\mathrm{leaves}}) + 1$")
    ax2.set_ylabel(r"$\log_{\,10}(\bar{t}_\mathrm{kernel}/1 \mathrm{ms})$")
    ax2.set_xlim(min(result.levels), max(result.levels))
    ax2.set_ylim(np.log10(np.amin(result.timings)),
                 np.log10(np.amax(result.timings)))
    ax2.legend(loc="upper left")

plt.show()
