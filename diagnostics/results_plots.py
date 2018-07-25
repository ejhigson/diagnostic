#!/usr/bin/env python
"""Functions for making the plots in the paper."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import getdist
import getdist.plots
import nestcheck.ns_run_utils
import nestcheck.estimators as e
import diagnostics.results_utils


def getdist_plot(run_list, **kwargs):
    """
    Makes a triangle plot of the nested sampling runs using getdist.

    N.B. for getdist to work with matplotlib usetex=True we need curly
    brackets around the subscript in e.g. \\theta_{\\hat{1}}"""
    width_inch = kwargs.pop('width_inch', 1)
    params = kwargs.pop('params', None)
    param_limits = kwargs.pop('param_limits', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    samples_list = []
    labels = diagnostics.results_utils.param_list_given_dim(
        run_list[0]['theta'].shape[1])
    # Strip the $s as getdist adds these
    labels = [lab.replace('$', '') for lab in labels]
    if params is None:
        params = labels
    elif isinstance(params, int):
        params = labels[:params]
    for i, run in enumerate(run_list):
        logw = nestcheck.ns_run_utils.get_logw(run)
        weights = np.exp(logw - logw.max())
        weights /= np.sum(weights)
        # remove zero weights as they can throw errors
        inds = np.nonzero(weights)
        samples_list.append(getdist.MCSamples(
            samples=run['theta'][inds, :],
            names=labels, weights=weights[inds], labels=labels,
            label='Run ' + str(i + 1)))
    gplot = getdist.plots.getSubplotPlotter(width_inch=width_inch)
    run_colors = ['red', 'blue']
    gplot.triangle_plot(
        samples_list, params=params, param_limits=param_limits,
        contour_colors=run_colors,
        diag1d_kwargs={'normalized': True},
        line_args=[{'color': col} for col in run_colors])
    return gplot


def ratio_bar_plot(errors_df, figsize=(3, 1)):
    """Bar plot of implementation error fractions."""
    ratio_plot = errors_df.xs(
        'implementation std frac', level='calculation type').rename(
            index={'LogGamma mix': 'LogGamma'})
    ratio_plot = ratio_plot.reorder_levels([1, 0]).T
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ratio_plot['value'].plot.bar(yerr=ratio_plot['uncertainty'], ax=ax,
                                 label='hello')
    ax.axhline(2 ** (-0.5), color='black', linestyle='dashed', linewidth=1,
               label=r'$\sigma_\mathrm{imp} = \sigma_\mathrm{bs}$')
    ax.set_ylim([0, 1])
    ax.set_ylabel(r'$\sigma_\mathrm{imp}$ / $\sigma_\mathrm{values}$',
                  labelpad=10)
    ax.legend(bbox_to_anchor=(1.02, 1))
    # Add line showing 1/sqrt(2)
    plt.xticks(rotation=0)
    return fig


def hist_plot(df_in, **kwargs):
    """Make a histogram plot of the dataframe values.

    Dataframe must have two index levels, with the first level determining the
    rows of plots and the second level the values to plot on each."""
    assert df_in.index.nlevels == 2, df_in.index.nlevels
    xlims = {'thread ks pvalue': [0, 1],
             'thread ks distance': [0, 0.3],
             'thread earth mover distance': [0, 0.2],
             'thread energy distance': [0, 0.4],
             'bootstrap ks pvalue': [0, 1],
             'bootstrap ks distance': [0, 1],
             'bootstrap earth mover distance': [0, 0.25],
             'bootstrap energy distance': [0, 0.8]}
    xlims = kwargs.get('xlims', xlims)
    figsize = kwargs.pop('figsize', (6.4, 2.5))
    nbin = kwargs.pop('nbin', 50)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    fig, axes = plt.subplots(
        nrows=len(set(df_in.index.get_level_values(0))),
        ncols=df_in.shape[1], sharex=True, sharey=True, figsize=figsize)
    plt.subplots_adjust(wspace=0)
    nax = 0
    for nr, row_name in enumerate(set(df_in.index.get_level_values(0))):
        df = df_in.xs(row_name, level=0)
        for i, (col_name, col) in enumerate(df.iteritems()):
            nax = i + nr * df_in.shape[1]
            try:
                ax = axes[nax]
            except TypeError:
                if df_in.shape[1] == 1:
                    ax = axes
                else:
                    raise
            # Need to set range and number of bins so all plots have same bin
            # sizes and frequencies can be compared
            col.plot.hist(ax=ax, range=xlims[row_name], bins=nbin,
                          label=col_name, alpha=0.7)
            # Add lines for quantiles
            ax.axvline(col.median(), ymax=0.65, color='black',
                       linestyle='dashed', linewidth=1)
            ax.set_title(col_name, y=0.6)
            ax.set_xlim(xlims[row_name])
            lab = (row_name.replace('thread ', '')
                   .replace('bootstrap ', '').title()
                   .replace('Ks', 'KS').replace('Pvalue', '$p$-value'))
            ax.set_xlabel(lab)
            # remove overlappling labels on x axis
            if i != df.shape[1] - 1:
                if (plt.gca().xaxis.get_majorticklocs()[-1] ==
                        xlims[row_name][1]):
                    xticks = ax.xaxis.get_major_ticks()
                    xticks[-1].label1.set_visible(False)
            ax.tick_params(axis='y', direction='inout')
    return fig


def get_line_plot(df_temp, estimator_name, figsize=(1.5, 3)):
    """Make line plots."""
    x_label_map = {'ndim': 'number of dimensions $d$',
                   'nlive': r'{\sc PolyChord} number of live points',
                   'nrepeats': r'{\sc PolyChord} \texttt{num\_repeats}'}
    xaxis_name = [name for name in df_temp.index.names if name
                  in x_label_map.keys()]
    assert len(xaxis_name) == 1, df_temp.index.names
    xaxis_name = xaxis_name[0]
    likelihood_list = list(set(df_temp.index.get_level_values('likelihood')))
    linestyles = ['-', '--', ':', '-.']
    # Make the plot
    fig, axes = plt.subplots(nrows=len(likelihood_list), ncols=1,
                             sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0)
    for nlike, likelihood_name in enumerate(likelihood_list):
        ax = axes[nlike]
        for nc, calc in enumerate(['values std', 'bootstrap std mean',
                                   'implementation std']):
            ser = (df_temp.xs(likelihood_name, level='likelihood')
                   .xs(calc, level='calculation type'))[estimator_name]
            ser = ser.sort_index()
            ser.xs('value', level='result type').plot.line(
                yerr=ser.xs('uncertainty', level='result type'),
                ax=ax, label=calc, linestyle=linestyles[nc])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if (xaxis_name == 'nlive' and likelihood_name == 'Gaussian'
                and estimator_name == e.get_latex_name(e.param_mean)):
            ax.set_yticks([0, 0.05, 0.1])
        if xaxis_name != 'ndim':
            ax.set_xscale('log')
        ax.set_ylabel('St.Dev.')
        title = (likelihood_name.title().replace('_', ' ')
                 .replace('Shell', 'shell'))
        title += ' ' + estimator_name
        ax.set_title(title, y=0.72)
        # make sure the labels of plots above and below each other don't clash
        ax.set_ylim([0, ax.get_yticks()[-1]])
        ax.tick_params(top=True, direction='inout')
        if nlike != 0:
            labels = ax.get_yticks().tolist()
            ax.set_yticks(labels[:-1])
        if nlike == len(likelihood_list) - 1:
            ax.set_xlabel(x_label_map[ax.get_xlabel()])
    return fig
