#!/usr/bin/env python
"""Functions for making the plots in the paper."""
import functools
from more_itertools import unique_everseen
import numpy as np
import scipy.integrate
import nestcheck.estimators as e
import dyPolyChord.python_likelihoods as likelihoods


# Settings
# --------


def get_default_lims(like_name, ndim_max=20):
    """Get some default param limits for the likelihoods used in the paper."""
    dim_labels = param_list_given_dim(ndim_max)
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


# Estimators
# ----------


def component_value(theta, ind=0):
    """Get parameter value from theta array given component index."""
    try:
        return theta[:, ind]
    except IndexError:
        assert theta.ndim == 2
        return np.full(theta[0], np.nan)


def radius(theta):
    """Function which gets the radial coordinates given a theta array in which
    each row is a position."""
    return np.sqrt(np.sum(theta ** 2, axis=1))


def get_ftheta_list(labels_in, ndim_max=20):
    """Get a list of ftheta functions.

    Each ftheta maps a 2d theta array to the function value for each row."""
    ftheta_dict = {r'$|\theta|$': radius}
    for i, lab in enumerate(param_list_given_dim(ndim_max)):
        ftheta_dict[lab] = functools.partial(component_value, ind=i)
    return [ftheta_dict[lab] for lab in labels_in]


# Names
# -----


def param_latex_name(i):
    """Param latex name. Numbered starting at 1.

    N.B. the curley braces around the subscript are needed as without them
    gedist throws an error."""
    assert i > 0, i
    return r'$\theta_{{\hat{{{}}}}}$'.format(i)


def param_list_given_dim(ndim):
    """List of param names."""
    return [param_latex_name(i) for i in range(1, ndim + 1)]


# Analytic values
# ---------------


def get_true_values_dict(ndim=10, prior_scale=30,
                         likelihood_list=['Gaussian', 'LogGamma mix']):
    """Calculate the correct values for esimaters numerically."""
    tv_dict = {}
    for like in likelihood_list:
        tv_dict[like] = {}
        if like == 'Gaussian':
            tv_dict[like][e.get_latex_name(e.logz)] = (
                ndim * -np.log(2 * prior_scale))
            tv_dict[like][e.get_latex_name(e.evidence)] = (
                (2 * prior_scale) ** (-ndim))
            for i in range(ndim):
                tv_dict[like][e.get_latex_name(
                    e.param_mean, param_ind=i)] = 0
        elif like == 'LogGamma mix':
            tv_dict[like][e.get_latex_name(e.logz)] = (
                ndim * -np.log(2 * prior_scale))
            tv_dict[like][e.get_latex_name(e.evidence)] = (
                (2 * prior_scale) ** (-ndim))
            loggamma_mean = scipy.integrate.quad(
                lambda x: x * np.exp(likelihoods.log_loggamma_pdf_1d(x)),
                -prior_scale, prior_scale)[0]
            boundry = 1 + (ndim // 2)
            for i in range(ndim):
                lab = e.get_latex_name(e.param_mean, param_ind=i)
                if i == 1 or i >= boundry:
                    tv_dict[like][lab] = 0
                else:
                    tv_dict[like][lab] = loggamma_mean
    return tv_dict


# def get_true_values_dict(prior_scale=10):
#     """Numberically calculate correct values for esimators."""
#     tv_fthetas = {}
#     tv_fthetas[e.get_latex_name(e.param_mean)] = lambda x: x[0]
#     tv_fthetas[e.get_latex_name(
#         functools.partial(e.param_mean, param_ind=1))] = lambda x: x[1]
#     tv_fthetas[e.get_latex_name(
#         e.param_squared_mean)] = lambda x: x[0] ** 2
#     tv_fthetas[e.get_latex_name(functools.partial(
#         e.param_squared_mean, param_ind=1))] = lambda x: x[1] ** 2
#     tv_fthetas[e.get_latex_name(
#         e.r_mean)] = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
#     true_values_dict = {}
#     options = {"epsabs": 1.49e-11, "epsrel": 1.49e-11, 'limit': 5000}
#     for like in [likelihoods.Gaussian(), likelihoods.GaussianShell(),
#                  likelihoods.Rastrigin(), likelihoods.Rosenbrock()]:
#         name = type(like).__name__.replace('GaussianShell', 'Gaussian shell')
#         print(name)
#         z = scipy.integrate.nquad(integrand_z,
#                                   ranges=[(-prior_scale, prior_scale),
#                                           (-prior_scale, prior_scale)],
#                                   args=(like,), opts=options)
#         true_values_dict[name] = {}
#         true_values_dict[name][e.get_latex_name(e.logz)] = np.log(z[0])
#         for ftheta_name, ftheta in tv_fthetas.items():
#             val = scipy.integrate.nquad(integrand_func_not_normed,
#                                         ranges=[(-prior_scale, prior_scale),
#                                                 (-prior_scale, prior_scale)],
#                                         args=(like, ftheta), opts=options)
#             true_values_dict[name][ftheta_name] = val[0] / z[0]
#     return true_values_dict
#
#
# def integrand_z(x2, x1, pc_like, prior_scale=10):
#     return (np.exp(pc_like(np.asarray([x1, x2]))[0])
#             / ((2 * prior_scale) ** 2))
#
#
# def integrand_func_not_normed(x2, x1, pc_like, ftheta, prior_scale=10):
#     return (ftheta((x1, x2)) * np.exp(pc_like(np.asarray([x1, x2]))[0])
#             / ((2 * prior_scale) ** 2))
