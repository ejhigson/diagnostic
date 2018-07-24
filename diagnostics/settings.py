#!/usr/bin/env python
"""Functions for making the plots in the paper."""
from more_itertools import unique_everseen
import diagnostics.results_utils


def get_default_lims(like_name, ndim=20):
    """Get some default param limits for the likelihoods used in the paper."""
    dim_labels = diagnostics.results_utils.param_list_given_dim(ndim)
    gaussian_lims = [-4, 4]  # define as used in multiple likelihoods
    lims = {}
    if like_name in ['LogGamma mix', 'LogGammaMix']:
        lims[dim_labels[0]] = [-20, 20]
        lims[dim_labels[1]] = [-20, 20]
        lims[r'$|\theta|$'] = [10, 20]
        if len(dim_labels) > 2:
            assert len(dim_labels) % 2 == 0, len(dim_labels)
            boundry = (len(dim_labels) // 2) + 1
            for i in range(2, len(dim_labels)):
                if i <= boundry:
                    # LogGamma
                    lims[dim_labels[i]] = [-7, 3]
                else:
                    # Gaussian
                    lims[dim_labels[i]] = gaussian_lims
    elif like_name == 'Gaussian':
        for lab in dim_labels:
            lims[lab] = gaussian_lims
        lims[r'$|\theta|$'] = [0, 6]
    else:
        raise AssertionError(
            'likename={} does not have default limits'.format(like_name))
    return lims


def default_logx_min(likelihood_name, ndim):
    """Default parameter for logx_min in diagrams."""
    return -ndim * 4


def get_default_nd_nl_nr():
    """Default values for dim, nlive and num_repeats - the paper individually
    varies each while holding the other two constant."""
    return (10, 200, 10)


def get_nd_nl_nr_list(**kwargs):
    """Get list of (dim, nlive, nrepeats) tuples."""
    defaults = get_default_nd_nl_nr()
    nd_list = kwargs.pop('nd_list', [2, 4, 10])
    nl_list = kwargs.pop('nl_list', [10, 20, 50, 200, 500, 1000])
    nr_list = kwargs.pop('nr_list', [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    nd_nl_nr_list = []
    for nd in nd_list:
        nd_nl_nr_list.append((nd, defaults[1], defaults[2]))
    for nl in nl_list:
        nd_nl_nr_list.append((defaults[0], nl, defaults[2]))
    for nr in nr_list:
        nd_nl_nr_list.append((defaults[0], defaults[1], nr))
    return list(unique_everseen(nd_nl_nr_list))