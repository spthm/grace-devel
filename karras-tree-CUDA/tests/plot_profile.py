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
            raise IOError("File " + fname + " exists but cannot be opened")
        else:
            break


for (i,result) in enumerate(profile_results):
    fig = plt.figure(i)
    ax = fig.add_subplot(111)
    for kernel in result.kernel_IDs:
        if not kernel == result.fastest_kernel_ID:
            ax.plot(result.levels, np.log10(result.timings[kernel]),
                    label=result.kernel_name_map[kernel],
                    linestyle='--', linewidth=0.2)
        else:
            ax.plot(result.levels, np.log10(result.timings[kernel]),
                    label=result.kernel_name_map[kernel])
    ax.set_xlabel(r"$\log_{\,2} (N_{\mathrm{leaves}}) + 1 = " +
                  r"\log_{\, 2} (N_{\mathrm{leaves}} N_{\mathrm{nodes}} + 1)$")
    ax.set_ylabel(r"$\log_{\,10}(\bar{\tau}_\mathrm{kernel}/1 \mathrm{ms})" +
                  r"\; (" + str(result.params['iterations']) +
                  r"\; \mathrm{ iterations})$")
    ax.legend()
plt.show()
