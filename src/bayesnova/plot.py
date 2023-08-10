import copy
import matplotlib
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as pl
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    NullLocator,
    ScalarFormatter,
)
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, entropy

def _parse_input(xs):
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    return xs

def _get_fig_axes(fig, K):
    if not fig.axes:
        return fig.subplots(K, K), True
    try:
        return np.array(fig.axes).reshape((K, K)), False
    except ValueError:
        raise ValueError(
            (
                "Provided figure has {0} axes, but data has "
                "dimensions K={1}"
            ).format(len(fig.axes), K)
        )

def quantile(a, q, w=None, interpolation='linear'):
    """Compute the weighted quantile for a one dimensional array."""
    if w is None:
        w = np.ones_like(a)
    a = np.array(list(a))  # Necessary to convert pandas arrays
    w = np.array(list(w))  # Necessary to convert pandas arrays
    i = np.argsort(a)
    c = np.cumsum(w[i[1:]]+w[i[:-1]])
    c = c / c[-1]
    c = np.concatenate(([0.], c))
    icdf = interp1d(c, a[i], kind=interpolation)
    quant = icdf(q)
    if isinstance(q, float):
        quant = float(quant)
    return quant

def quantile_plot_interval(q):
    """Interpret quantile ``q`` input to quantile plot range tuple."""
    if isinstance(q, str):
        sigmas = {'1sigma': 0.682689492137086,
                  '2sigma': 0.954499736103642,
                  '3sigma': 0.997300203936740,
                  '4sigma': 0.999936657516334,
                  '5sigma': 0.999999426696856}
        q = (1 - sigmas[q]) / 2
    elif isinstance(q, int) and q >= 1:
        q = (1 - erf(q / np.sqrt(2))) / 2
    if isinstance(q, float) or isinstance(q, int):
        if q > 0.5:
            q = 1 - q
        q = (q, 1-q)
    return tuple(np.sort(q))

def cut_and_normalise_gaussian(x, p, bw, xmin=None, xmax=None):
    """Cut and normalise boundary correction for a Gaussian kernel.

    Parameters
    ----------
    x : array-like
        locations for normalisation correction

    p : array-like
        probability densities for normalisation correction

    bw : float
        bandwidth of KDE

    xmin, xmax : float
        lower/upper prior bound
        optional, default None

    Returns
    -------
    p : np.array
        corrected probabilities

    """
    correction = np.ones_like(x)

    if xmin is not None:
        correction *= 0.5 * (1 + erf((x-xmin)/bw/np.sqrt(2)))
        correction[x < xmin] = np.inf
    if xmax is not None:
        correction *= 0.5 * (1 + erf((xmax-x)/bw/np.sqrt(2)))
        correction[x > xmax] = np.inf
    return p/correction

def iso_probability_contours(pdf, contours=[0.95, 0.68]):
    """Compute the iso-probability contour values."""
    if len(contours) > 1 and not np.all(contours[:-1] > contours[1:]):
        raise ValueError(
            "The kwargs `levels` and `contours` have to be ordered from "
            "outermost to innermost contour, i.e. in strictly descending "
            "order when referring to the enclosed probability mass, e.g. "
            "like the default [0.95, 0.68]. "
            "This breaking change in behaviour was introduced in version "
            "2.0.0-beta.10, in order to better match the ordering of other "
            "matplotlib kwargs."
        )
    contours = [1-p for p in contours]
    p = np.sort(np.array(pdf).flatten())
    m = np.cumsum(p)
    m /= m[-1]
    interp = interp1d([0]+list(m), [0]+list(p))
    c = list(interp(contours))+[max(p)]

    return c

def basic_cmap(color):
    """Construct basic colormap a single color."""
    return LinearSegmentedColormap.from_list(str(color), ['#ffffff', color])

def kde_plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d marginalised distribution.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.plot`, with
    a kernel density estimation computation provided by
    :class:`scipy.stats.gaussian_kde` in-between. All remaining keyword
    arguments are passed onwards.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data : np.array
        Samples to generate kernel density estimator.

    weights : np.array, optional
        Sample weights.

    ncompress : int, str, default=False
        Degree of compression.

        * If ``False``: no compression.
        * If ``True``: compresses to the channel capacity, equivalent to
          ``ncompress='entropy'``.
        * If ``int``: desired number of samples after compression.
        * If ``str``: determine number from the Huggins-Roy family of
          effective samples in :func:`anesthetic.utils.neff`
          with ``beta=ncompress``.

    nplot_1d : int, default=100
        Number of plotting points to use.

    levels : list
        Values at which to draw iso-probability lines.
        Default: [0.95, 0.68]

    q : int or float or tuple, default=5
        Quantile to determine the data range to be plotted.

        * ``0``: full data range, i.e. ``q=0`` --> quantile range (0, 1)
        * ``int``: q-sigma range, e.g. ``q=1`` --> quantile range (0.16, 0.84)
        * ``float``: percentile, e.g. ``q=0.8`` --> quantile range (0.1, 0.9)
        * ``tuple``: quantile range, e.g. (0.16, 0.84)

    facecolor : bool or string, default=False
        If set to True then the 1d plot will be shaded with the value of the
        ``color`` kwarg. Set to a string such as 'blue', 'k', 'r', 'C1' ect.
        to define the color of the shading directly.

    bw_method : str, scalar or callable, optional
        Forwarded to :class:`scipy.stats.gaussian_kde`.

    beta : int, float, default = 1
        The value of beta used to calculate the number of effective samples

    Returns
    -------
    lines : :class:`matplotlib.lines.Line2D`
        A list of line objects representing the plotted data (same as
        :meth:`matplotlib.axes.Axes.plot` command).

    """

    weights = kwargs.pop('weights', None)
    if weights is not None:
        data = data[weights != 0]
        weights = weights[weights != 0]

    nplot = kwargs.pop('nplot_1d', 100)
    bw_method = kwargs.pop('bw_method', None)
    levels = kwargs.pop('levels', [0.95, 0.68])
    density = kwargs.pop('density', False)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None
                                 else plt.get_cmap(cmap)(0.68)))
    facecolor = kwargs.pop('facecolor', False)
    if 'edgecolor' in kwargs:
        edgecolor = kwargs.pop('edgecolor')
        if edgecolor:
            color = edgecolor
    else:
        edgecolor = color
    
    linewidth = kwargs.pop('linewidth', 3)

    q = kwargs.pop('q', 5)
    range_override = kwargs.pop('range_override', None)
    if range_override is not None:
        xmin, xmax = range_override
    else:
        q = quantile_plot_interval(q=q)
        xmin = quantile(data, q[0], weights)
        xmax = quantile(data, q[-1], weights)
    x = np.linspace(xmin, xmax, nplot)

    data_compressed, w = data, weights
    kde = gaussian_kde(data_compressed, weights=w, bw_method=bw_method)

    p = kde(x)
    p /= p.max()
    bw = np.sqrt(kde.covariance[0, 0])
    pp = cut_and_normalise_gaussian(x, p, bw, xmin=data.min(), xmax=data.max())
    pp /= pp.max()
    area = np.trapz(x=x, y=pp) if density else 1
    ans = ax.plot(x, pp/area, color=color, lw=linewidth, *args, **kwargs)

    if facecolor and facecolor not in [None, 'None', 'none']:
        if facecolor is True:
            facecolor = color
        c = iso_probability_contours(pp, contours=levels)
        cmap = basic_cmap(facecolor)
        fill = []
        for j in range(len(c)-1):
            fill.append(ax.fill_between(x, pp, where=pp >= c[j],
                        color=cmap(c[j]), edgecolor=edgecolor))

        ans = ans, fill

    if density:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, 1.1)

    return ans

def set_colors(c, fc, ec, cmap):
    """Navigate interplay between possible color inputs {c, fc, ec, cmap}."""
    if fc in [None, 'None', 'none']:
        # unfilled contours
        if ec is None and cmap is None:
            cmap = basic_cmap(c)
    else:
        # filled contours
        if fc is True:
            fc = c
        if ec is None and cmap is None:
            ec = c
            cmap = basic_cmap(fc)
        elif ec is None:
            ec = (cmap(1.),)
        elif cmap is None:
            cmap = basic_cmap(fc)
    return fc, ec, cmap

def neff(w, beta=1):
    r"""Calculate effective number of samples.

    Using the Huggins-Roy family of effective samples
    (https://aakinshin.net/posts/huggins-roy-ess/).

    Parameters
    ----------
    beta : int, float, str, default = 1
        The value of beta used to calculate the number of effective samples
        according to

        .. math::

            N_{eff} &= \bigg(\sum_{i=0}^n w_i^\beta \bigg)^{\frac{1}{1-\beta}}

            w_i &= \frac{w_i}{\sum_j w_j}

        Beta can take any positive value. Larger beta corresponds to a greater
        compression such that:

        .. math::

            \beta_1 < \beta_2 \Rightarrow N_{eff}(\beta_1) > N_{eff}(\beta_2)

        Alternatively, beta can take one of the following strings as input:

        * If 'inf' or 'equal' is supplied (equivalent to beta=inf), then the
          resulting number of samples is the number of samples when compressed
          to equal weights, and given by:

        .. math::

            w_i &= \frac{w_i}{\sum_j w_j}

            N_{eff} &= \frac{1}{\max_i[w_i]}

        * If 'entropy' is supplied (equivalent to beta=1), then the estimate
          is determined via the entropy based calculation, also referred to as
          the channel capacity:

        .. math::

            H &= -\sum_i p_i \ln p_i

            p_i &= \frac{w_i}{\sum_j w_j}

            N_{eff} &= e^{H}

        * If 'kish' is supplied (equivalent to beta=2), then a Kish estimate
          is computed (Kish, Leslie (1965). Survey Sampling.
          New York: John Wiley & Sons, Inc. ISBN 0-471-10949-5):

        .. math::

            N_{eff} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}

        * str(float) input gets converted to the corresponding float value.

    """
    w = w / np.sum(w)
    if beta == np.inf or beta == 'inf' or beta == 'equal':
        return 1 / np.max(w)
    elif beta == 'entropy' or beta != 'kish' and str(float(beta)) == '1.0':
        return np.exp(entropy(w))
    else:
        if beta == 'kish':
            beta = 2
        elif isinstance(beta, str):
            beta = float(beta)
        return np.sum(w**beta)**(1/(1-beta))

def scaled_triangulation(x, y, cov):
    """Triangulation scaled by a covariance matrix.

    Parameters
    ----------
    x, y : array-like
        x and y coordinates of samples

    cov : array-like, 2d
        Covariance matrix for scaling

    Returns
    -------
    :class:`matplotlib.tri.Triangulation`
        Triangulation with the appropriate scaling
    """
    L = np.linalg.cholesky(cov)
    Linv = np.linalg.inv(L)
    x_, y_ = Linv.dot([x, y])
    tri = Triangulation(x_, y_)
    return Triangulation(x, y, tri.triangles)

def triangular_sample_compression_2d(x, y, cov, w=None, n=1000):
    """Histogram a 2D set of weighted samples via triangulation.

    This defines bins via a triangulation of the subsamples and sums weights
    within triangles surrounding each point

    Parameters
    ----------
    x, y : array-like
        x and y coordinates of samples for compressing

    cov : array-like, 2d
        Covariance matrix for scaling

    w : :class:`pandas.Series`, optional
        weights of samples

    n : int, default=1000
        number of samples returned.

    Returns
    -------
    tri :
        :class:`matplotlib.tri.Triangulation` with an appropriate scaling

    w : array-like
        Compressed samples and weights
    """
    # Pre-process samples to not be affected by non-standard indexing
    # Details: https://github.com/handley-lab/anesthetic/issues/189
    x = np.array(x)
    y = np.array(y)

    x = pd.Series(x)
    if w is None:
        w = pd.Series(index=x.index, data=np.ones_like(x))

    if isinstance(n, str):
        n = int(neff(w, beta=n))

    # Select samples for triangulation
    if (w != 0).sum() < n:
        i = x.index
    else:
        i = np.random.choice(x.index, size=n, replace=False, p=w/w.sum())

    # Generate triangulation
    tri = scaled_triangulation(x[i], y[i], cov)

    # For each point find corresponding triangles
    trifinder = tri.get_trifinder()
    j = trifinder(x, y)
    k = tri.triangles[j[j != -1]]

    # Compute mass in each triangle, and add it to each corner
    w_ = np.zeros(len(i))
    for i in range(3):
        np.add.at(w_, k[:, i], w[j != -1]/3)

    return tri, w_

def match_contour_to_contourf(contours, vmin, vmax):
    """Get needed `vmin, vmax` to match `contour` colors to `contourf` colors.

    `contourf` uses the arithmetic mean of contour levels to assign colors,
    whereas `contour` uses the contour level directly. To get the same colors
    for `contour` lines as for `contourf` faces, we need some fiddly algebra.
    """
    if len(contours) <= 2:
        vmin = 2 * vmin - vmax
        return vmin, vmax
    else:
        c0 = contours[0]
        c1 = contours[1]
        ce = contours[-2]
        denom = vmax + ce - c1 - c0
        vmin = +(c0 * vmax - c1 * ce + 2 * vmin * (ce - c0)) / denom
        vmax = -(c0 * vmax + c1 * ce - 2 * vmax * ce) / denom
        return vmin, vmax

def kde_contour_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.contour`
    and :meth:`matplotlib.axes.Axes.contourf` with a kernel density
    estimation (KDE) computation provided by :class:`scipy.stats.gaussian_kde`
    in-between. All remaining keyword arguments are passed onwards to both
    functions.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data_x, data_y : np.array
        The x and y coordinates of uniformly weighted samples to generate
        kernel density estimator.

    weights : np.array, optional
        Sample weights.

    levels : list, optional
        Amount of mass within each iso-probability contour.
        Has to be ordered from outermost to innermost contour.
        Default: [0.95, 0.68]

    ncompress : int, str, default='equal'
        Degree of compression.

        * If ``int``: desired number of samples after compression.
        * If ``False``: no compression.
        * If ``True``: compresses to the channel capacity, equivalent to
          ``ncompress='entropy'``.
        * If ``str``: determine number from the Huggins-Roy family of
          effective samples in :func:`anesthetic.utils.neff`
          with ``beta=ncompress``.

    nplot_2d : int, default=1000
        Number of plotting points to use.

    bw_method : str, scalar or callable, optional
        Forwarded to :class:`scipy.stats.gaussian_kde`.

    Returns
    -------
    c : :class:`matplotlib.contour.QuadContourSet`
        A set of contourlines or filled regions.

    """

    weights = kwargs.pop('weights', None)
    if weights is not None:
        data_x = data_x[weights != 0]
        data_y = data_y[weights != 0]
        weights = weights[weights != 0]

    ncompress = kwargs.pop('ncompress', 'equal')
    nplot = kwargs.pop('nplot_2d', 1000)
    bw_method = kwargs.pop('bw_method', None)
    label = kwargs.pop('label', None)
    zorder = kwargs.pop('zorder', 1)
    levels = kwargs.pop(
        'levels',
        [
            0.989,
            0.865,
            0.393
        ]
    )

    alpha = kwargs.pop('alpha', 0.5)
    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    facecolor = kwargs.pop('facecolor', True)
    edgecolor = kwargs.pop('edgecolor', None)
    cmap = kwargs.pop('cmap', None)
    facecolor, edgecolor, cmap = set_colors(c=color, fc=facecolor,
                                            ec=edgecolor, cmap=cmap)
    linewidths = kwargs.pop('linewidth', 3)
    kwargs.pop('q', None)

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    range_override_x = kwargs.pop('range_override_x', None)
    range_override_y = kwargs.pop('range_override_y', None)
    if range_override_x is not None:
        xmin, xmax = range_override_x
    else:
        xmin = quantile(data_x, q[0], weights)
        xmax = quantile(data_x, q[-1], weights)
    if range_override_y is not None:
        ymin, ymax = range_override_y
    else:
        ymin = quantile(data_y, q[0], weights)
        ymax = quantile(data_y, q[-1], weights)
    X, Y = np.mgrid[xmin:xmax:1j*np.sqrt(nplot), ymin:ymax:1j*np.sqrt(nplot)]

    cov = np.cov(data_x, data_y, aweights=weights)
    tri, w = triangular_sample_compression_2d(data_x, data_y, cov,
                                              weights, ncompress)
    kde = gaussian_kde([tri.x, tri.y], weights=w, bw_method=bw_method)

    P = kde([X.ravel(), Y.ravel()]).reshape(X.shape)

    bw_x = np.sqrt(kde.covariance[0, 0])
    P = cut_and_normalise_gaussian(X, P, bw=bw_x,
                                   xmin=data_x.min(), xmax=data_x.max())
    bw_y = np.sqrt(kde.covariance[1, 1])
    P = cut_and_normalise_gaussian(Y, P, bw=bw_y,
                                   xmin=data_y.min(), xmax=data_y.max())

    levels = iso_probability_contours(P, contours=levels)

    if facecolor not in [None, 'None', 'none']:
        # linewidths = kwargs.pop('linewidths', 0.5)
        contf = ax.contourf(X, Y, P, levels=levels, cmap=cmap, zorder=zorder,
                            vmin=0, vmax=P.max(), alpha=alpha, *args, **kwargs)
        for c in contf.collections:
            c.set_cmap(cmap)
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                                   fc=cmap(0.999), ec=cmap(0.32)))
        cmap = None
    else:
        linewidths = kwargs.pop('linewidths',
                                plt.rcParams.get('lines.linewidth'))
        contf = None
        ax.add_patch(
            plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                          fc='None' if cmap is None else cmap(0.999),
                          ec=edgecolor if cmap is None else cmap(0.32))
        )

    if not isinstance(edgecolor, str):
        edgecolor = [edgecolor] 
    vmin, vmax = match_contour_to_contourf(levels, vmin=0, vmax=P.max())
    cont = ax.contour(X, Y, P, levels=levels, zorder=zorder,
                      vmin=vmin, vmax=vmax, linewidths=linewidths,
                      colors=edgecolor, cmap=cmap,)# *args, **kwargs)

    return contf, cont

def corner(
    xs,
    weights=None,
    color=None,
    labels=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    quantiles=None,
    title_quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    params_to_skip=[],
    hist2d_kwargs=None,
    range_overrides=None,
):
    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()

    # If no separate titles are set, copy the axis labels
    if titles is None:
        titles = labels

    # deal with title quantiles so they much quantiles unless desired otherwise
    if title_quantiles is None:
        if len(quantiles) > 0:
            title_quantiles = quantiles
        else:
            # a default for when quantiles not supplied.
            title_quantiles = [0.16, 0.5, 0.84]

    if show_titles and len(title_quantiles) != 3:
        raise ValueError(
            "'title_quantiles' must contain exactly three values; "
            "pass a length-3 list or array using the 'title_quantiles' argument"
        )

    # Deal with 1D sample lists.
    xs = _parse_input(xs)
    assert xs.shape[0] <= xs.shape[1], (
        "I don't believe that you want more " "dimensions than samples!"
    )

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0  # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor  # size of left/bottom margin
        trdim = 0.5 * factor  # size of top/right margin
    else:
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        axes, new_fig = _get_fig_axes(fig, K)

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )
    #     raise ValueError("Dimension mismatch between samples and range")

    # Set up the default plotting arguments.
    if color is None:
        color = matplotlib.rcParams["ytick.color"]

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)

    if range_overrides is None:
        range_overrides = {}

    for i, x in enumerate(xs):

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            if reverse:
                ax = axes[K - i - 1, K - i - 1]
            else:
                ax = axes[i, i]

        # Plot the histograms.
        range_override = range_overrides.get(i, None)
        skip_param = i in params_to_skip
        
        if not skip_param:
            kde_plot_1d(
                ax, x, weights=weights,
                range_override=range_override,
                **hist_kwargs
            )

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if show_titles:
            title = None
            if title_fmt is not None:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_lo, q_mid, q_hi = quantile(
                    x, title_quantiles, weights=weights
                )
                q_m, q_p = q_mid - q_lo, q_hi - q_mid

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_mid), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if titles is not None:
                    title = "{0} = {1}".format(titles[i], title)

            elif titles is not None:
                title = "{0}".format(titles[i])

            if title is not None:
                if reverse:
                    if "pad" in title_kwargs.keys():
                        title_kwargs_new = copy.copy(title_kwargs)
                        del title_kwargs_new["pad"]
                        title_kwargs_new["labelpad"] = title_kwargs["pad"]
                    else:
                        title_kwargs_new = title_kwargs

                    ax.set_xlabel(title, **title_kwargs_new)
                else:
                    ax.set_title(title, **title_kwargs)

        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(
                MaxNLocator(max_n_ticks, prune="lower")
            )
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                [l.set_rotation(45) for l in ax.get_xticklabels(minor=True)]
            else:
                ax.set_xticklabels([])
                ax.set_xticklabels([], minor=True)
        else:
            if reverse:
                ax.xaxis.tick_top()
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            [l.set_rotation(45) for l in ax.get_xticklabels(minor=True)]
            if labels is not None:
                if reverse:
                    if "labelpad" in label_kwargs.keys():
                        label_kwargs_new = copy.copy(label_kwargs)
                        del label_kwargs_new["labelpad"]
                        label_kwargs_new["pad"] = label_kwargs["labelpad"]
                    else:
                        label_kwargs_new = label_kwargs
                    ax.set_title(
                        labels[i],
                        position=(0.5, 1.3 + labelpad),
                        **label_kwargs_new,
                    )

                else:
                    ax.set_xlabel(labels[i], **label_kwargs)
                    ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text)
            )

        if hist_kwargs is None:
            hist_kwargs = dict()
        hist2d_kwargs["color"] = hist2d_kwargs.get("color", color)
        for j, y in enumerate(xs):

            # if skip_param:
            #     inverted_params_to_skip = set(range(len(xs))) - set(params_to_skip)
            #     if j in inverted_params_to_skip:
            #         continue
            if i in params_to_skip:
                if j in params_to_skip:
                    continue
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                if reverse:
                    ax = axes[K - i - 1, K - j - 1]
                else:
                    ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue
            
            y_range_override = range_override
            x_range_override = range_overrides.get(j, None)

            kde_contour_plot_2d(
                ax=ax,
                data_x=y,
                data_y=x,
                weights=weights,
                range_override_x=x_range_override,
                range_override_y=y_range_override,
                **hist2d_kwargs
            )

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(
                        MaxNLocator(max_n_ticks, prune="lower")
                    )
                ax.yaxis.set_major_locator(
                        MaxNLocator(max_n_ticks, prune="lower")
                    )

            if i < K - 1:
                ax.set_xticklabels([])
                ax.set_xticklabels([], minor=True)
            else:
                if reverse:
                    ax.xaxis.tick_top()
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                [l.set_rotation(45) for l in ax.get_xticklabels(minor=True)]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4 + labelpad)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text)
                )

            if j > 0:
                ax.set_yticklabels([])
                ax.set_yticklabels([], minor=True)
            else:
                if reverse:
                    ax.yaxis.tick_right()
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                [l.set_rotation(45) for l in ax.get_yticklabels(minor=True)]
                if labels is not None:
                    if reverse:
                        ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3 + labelpad, 0.5)
                    else:
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3 - labelpad, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                        ScalarFormatter(useMathText=use_math_text)
                    )


    return fig