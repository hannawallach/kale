from numpy import asarray, linspace, shape, sqrt
from pylab import gca, figure, ion, setp, subplots


class InteractivePlot(object):
    """
    Interactive plot that can be updated with new (x, y) pairs.
    """

    def __init__(self, xlabel, ylabel):
        """
        Initializes plot.
        """

        self.xlabel, self.ylabel = xlabel, ylabel

        ion() # has to come before the next 2 lines

        self.fig = figure()
        self.axes = self.fig.gca()

        self.x_values, self.y_values = [], []

    def update(self, x, y):
        """
        Update this plot to include the specified (x, y) pair.

        Arguments:

        x -- x value
        y -- y value
        """

        x_values, y_values = self.x_values, self.y_values

        x_values.append(x)
        y_values.append(y)

        axes = self.axes

        axes.cla()

        axes.plot(x_values, y_values, 'k')

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)

        # have to call this twice (workaround for matplotlib weirdness)

        self.fig.canvas.draw()
        self.fig.canvas.draw()


def textplot(coords, sizes, labels,
             axes=None,
             limits=None,
             text_opts={}):
    """
    Creates a text plot.

    Arguments:

    coords --
    sizes --
    labels --

    Keyword arguments:

    axes --
    limits -- x and y axis limits
    text_opts -- additional arguments to text(...)
    """

    assert shape(coords) == (len(labels), 2) and shape(sizes) == (len(labels),)

    coords = asarray(coords, dtype=float)
    sizes = asarray(sizes, dtype=float)

    sizes = (sizes - sizes.min()) / sizes.max() * 20 + 8

    if axes is None:
        axes = gca()

    if limits is not None:
        axes.axis(limits)

    assert 'fontsize' not in text_opts and 'size' not in text_opts

    text_defaults = dict(ha='center')
    text_defaults.update(text_opts)

    for (x, y), size, s in zip(coords, sizes, labels):

        axes.text(x, y, s, size=size, **text_defaults)
        axes.plot(x, y, alpha=0.0)


def bubbleplot(coords, sizes, labels,
               axes=None,
               limits=None,
               text_opts={},
               scatter_opts={}):
    """
    Creates a bubble plot.

    Arguments:

    coords --
    sizes --
    labels --

    Keyword arguments:

    axes --
    limits -- x and y axis limits
    text_opts -- additional arguments to text(...)
    scatter_opts -- additional arguments to scatter(...)
    """

    assert shape(coords) == (len(labels), 2) and shape(sizes) == (len(labels),)

    coords = asarray(coords, dtype=float)
    sizes = asarray(sizes, dtype=float)

    if axes is None:
        axes = gca()

    if limits is not None:
        axes.axis(limits)

    text_defaults = dict(fontsize=11, ha='center')
    text_defaults.update(text_opts)

    for (x, y), s in zip(coords, labels):
        axes.text(x, y, s, **text_opts)

    scatter_defaults = dict(c='gray', alpha=0.5, linewidth=2, edgecolor='w')
    scatter_defaults.update(scatter_opts)

    axes.scatter(coords[:, 0], coords[:, 1], s=sqrt(sizes), **scatter_defaults)


def scatter_matrix(data, figsize=None, hist_opts={}, scatter_opts={}):
    """
    Creates a (lower diagonal) matrix of scatter plots, with
    histograms along the diagonal.

    Arguments:

    data --

    Keyword arguments:

    figsize --
    hist_opts -- additional arguments to hist(...)
    scatter_opts -- additional arguments to scatter(...)
    """

    data = asarray(data)

    assert len(data.shape) == 2

    num_rows = num_cols = data.shape[1]

    fig, axes = subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # subplots indexed from top left to bottom right
    # axes[0, 0] is top left, axes[num_rows, num_cols] is bottom right

    hist_defaults = dict(color='gray', linewidth=0)
    hist_defaults.update(hist_opts)

    scatter_defaults = dict(c='gray', alpha=0.4, linewidth=0)
    scatter_defaults.update(scatter_opts)

    for row in xrange(num_rows):
        for col in xrange(num_cols):

            ax = axes[row, col]

            xaxis = ax.get_xaxis()
            yaxis = ax.get_yaxis()

            if col > row:

                ax.axis('off')
                continue

            elif col == row:

                ax.hist(data[:, row], **hist_defaults)
                yaxis.set_visible(False)

            else:

                ax.scatter(data[:, col], data[:, row], **scatter_defaults)

                if col == 0:

                    ticks = linspace(data[:, row].min(), data[:, row].max(), 3)
                    labels = ['%.2g' % t for t in ticks]

                    yaxis.set_ticks(ticks)
                    yaxis.set_ticklabels(labels)
                else:
                    yaxis.set_visible(False)

            if row == num_rows - 1:

                ticks = linspace(data[:, col].min(), data[:, col].max(), 3)
                labels = ['%.2g' % t for t in ticks]

                xaxis.set_ticks(ticks)
                xaxis.set_ticklabels(labels)

            else:
                xaxis.set_visible(False)

            ax.grid(False)
            setp(xaxis.get_majorticklabels(), rotation=270)

    return fig, axes


if __name__ == '__main__':

    from sklearn import datasets
    from pylab import show

    scatter_matrix(datasets.load_diabetes().data)

    show()


#if __name__ == '__main__':

#    from text_utils import load_csv
#    from pylab import show

#    data, labels = [], []

#    for row in load_csv('data/crimeRatesByState2005.csv', header_rows=2):

#        data.append([row[i] for i in [1, 5, 8]])
#        labels.append(row[0])

#    data = asarray(data, dtype=float)

#    bubbleplot(data[:, 0:2], data[:, 2], labels)

#    show()


#if __name__ == '__main__':

#    from math import sin
#    from time import sleep

#    plt1 = InteractivePlot('x', 'x * sin(0.4 * x)')
#    plt2 = InteractivePlot('x', 'sin(0.4 * x)')

#    for x in xrange(100):

#        plt1.update(x, x * sin(0.4 * x))
#        plt2.update(x, sin(0.4 * x))

#        sleep(0.01)
