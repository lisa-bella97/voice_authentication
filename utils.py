import matplotlib.pyplot as plt


def stem(data, linefmt=None, markerfmt=None, basefmt='k', markersize=10):
    markerline, _, _ = plt.stem(data, linefmt=linefmt, markerfmt=markerfmt, basefmt=basefmt)
    markerline.set_markerfacecolor('none')
    plt.setp(markerline, markersize=markersize)
