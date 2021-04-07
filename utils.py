import os

import matplotlib.pyplot as plt


def stem(data, linefmt=None, markerfmt=None, basefmt='k', markersize=10):
    markerline, _, _ = plt.stem(data, linefmt=linefmt, markerfmt=markerfmt, basefmt=basefmt)
    markerline.set_markerfacecolor('none')
    plt.setp(markerline, markersize=markersize)


def recreate_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    open(filename, 'a').close()
