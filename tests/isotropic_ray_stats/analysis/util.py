import gc
from sys import stdout

import matplotlib.pyplot as plt

def print_overwrite(line):
    if len(line) < 80:
        line += ' ' * (80 - len(line))
    stdout.write('\r' + line)
    stdout.flush()

def print_overwrite_done():
    stdout.write('\r%s\r' % (" "*80, ))
    stdout.flush()

def release_fig(fig):
    fig.clf()
    plt.close()
    gc.collect()
