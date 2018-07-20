#!/usr/bin/env python
"""Functions for making the plots in the paper."""
import numpy as np
import matplotlib.pyplot as plt
import getdist
import getdist.plots
import nestcheck.ns_run_utils


def getdist_plot(run_list, lims, width_inch=1):
    """Makes a trianlge plot of the nested sampling runs using getdist."""
    samples_list = []
    labels = [r'$\theta_\hat{{{}}}$'.format(i) for i in
              range(1, run_list[0]['theta'].shape[1] + 1)]
    for i, run in enumerate(run_list):
        logw = nestcheck.ns_run_utils.get_logw(run)
        weights = np.exp(logw - logw.max())
        weights /= np.sum(weights)
        # remove zero weights as they can throw errors
        inds = np.nonzero(weights)
        samples_list.append(getdist.MCSamples(samples=run['theta'][inds, :],
                                              names=labels,
                                              weights=weights[inds],
                                              labels=labels,
                                              label='Run ' + str(i + 1)))
    gplot = getdist.plots.getSubplotPlotter(width_inch=width_inch)
    run_colors = ['red', 'blue']
    gplot.triangle_plot(
        samples_list, param_limits=lims,
        contour_colors=run_colors,
        diag1d_kwargs={'normalized': True},
        line_args=[{'color': col} for col in run_colors])
    return gplot


def ratio_bar_plot(errors_df, figsize=(3, 1)):
    """Bar plot of implementation error fractions."""
    ratio_plot = errors_df.xs('implementation std frac',
                              level='calculation type')
    ratio_plot = ratio_plot.reorder_levels([1, 0]).T
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ratio_plot['value'].plot.bar(yerr=ratio_plot['uncertainty'], ax=ax)
    # Add line showing 1/sqrt(2)
    ax.axhline(2 ** (-0.5), color='black',
               linestyle='dashed', linewidth=1)
    # ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_ylabel('Imp St.Dev. / Values St.Dev.', labelpad=10)
    ax.legend(bbox_to_anchor=(1.02, 1), title='Likelihood')
    plt.xticks(rotation=0)
    fig.subplots_adjust(left=0.11, right=0.7, bottom=0.14, top=0.97)
    return fig


def hist_plot(df_in, calculation, estimator, **kwargs):
    """Make a histogram plot of the dataframe values."""
    xlim = kwargs.get('xlim', None)
    figsize = kwargs.pop('figsize', (6.4, 2.5))
    nbin = kwargs.pop('nbin', 50)
    likelihood_list = kwargs.pop('likelihood_list', ['Gaussian'])
    fig, axes = plt.subplots(nrows=1, ncols=len(likelihood_list),
                             sharex=True, sharey=True, figsize=figsize)
    plt.subplots_adjust(wspace=0)
    for i, like in enumerate(likelihood_list):
        ax = axes[i]
        df = (df_in.xs(calculation, level='calculation type')
              .unstack(level='likelihood')[estimator])
        # Need to set range and number of bins so all plots have same bin
        # sizes and frequencies can be compared
        df[like].plot.hist(ax=ax, range=(xlim[0], xlim[1]), bins=nbin,
                           label=estimator, alpha=0.7)
        # Add lines for quantiles
        ax.axvline(df[like].median(), ymax=0.65, color='black',
                   linestyle='dashed', linewidth=1)
        ax.set_title(like.replace('Shell', 'shell') + ' ' + estimator, y=0.65)
        if xlim is not None:
            ax.set_xlim(xlim)
        lab = (calculation.replace('thread ', '')
               .replace('bootstrap ', '').title()
               .replace('Ks', 'KS').replace('Pvalue', '$p$-value'))
        ax.set_xlabel(lab)
        # remove overlappling labels on x axis
        if i != len(likelihood_list) - 1:
            if plt.gca().xaxis.get_majorticklocs()[-1] == xlim[1]:
                xticks = ax.xaxis.get_major_ticks()
                xticks[-1].label1.set_visible(False)
        ax.tick_params(axis='y', direction='inout')
    return fig
