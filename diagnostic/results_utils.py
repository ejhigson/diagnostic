#!/usr/bin/env python
"""Utilities for helping process paper results."""
import functools
import numpy as np
import scipy.integrate
import nestcheck.estimators as e
import dyPolyChord.python_likelihoods as likelihoods


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
    r"""Get a list of ftheta functions.

    Each ftheta maps a 2d theta array to the function value for each row.

    N.B. boldsymbol requires:

    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    If you can't load amsmath, just use |\theta| instead of
    |\boldsymbol{\theta}|.
    """
    ftheta_dict = {r'$|\boldsymbol{\theta}|$': radius,
                   r'$|\theta|$': radius}
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


def get_true_values(likelihood_name, ndim, estimator_names):
    """Array of true values with nans where they are not available."""
    tv_dict = get_true_values_dict(likelihood_name, ndim)
    true_values = np.full(len(estimator_names), np.nan)
    for i, name in enumerate(estimator_names):
        try:
            true_values[i] = tv_dict[name]
        except KeyError:
            pass
    return true_values


def get_true_values_dict(likelihood_name, ndim, prior_scale=30):
    """Calculate the correct values for esimaters numerically."""
    tv_dict = {}
    if likelihood_name == 'Gaussian':
        tv_dict[e.get_latex_name(e.logz)] = (
            ndim * -np.log(2 * prior_scale))
        tv_dict[e.get_latex_name(e.evidence)] = (
            (2 * prior_scale) ** (-ndim))
        for i in range(ndim):
            tv_dict[e.get_latex_name(
                e.param_mean, param_ind=i)] = 0
    elif likelihood_name == 'LogGamma mix':
        assert ndim % 2 == 0
        tv_dict[e.get_latex_name(e.logz)] = (
            ndim * -np.log(2 * prior_scale))
        tv_dict[e.get_latex_name(e.evidence)] = (
            (2 * prior_scale) ** (-ndim))
        loggamma_mean = scipy.integrate.quad(
            lambda x: x * np.exp(likelihoods.log_loggamma_pdf_1d(x)),
            -prior_scale, prior_scale)[0]
        boundry = 1 + (ndim // 2)
        for i in range(ndim):
            lab = e.get_latex_name(e.param_mean, param_ind=i)
            if i == 1 or i >= boundry:
                tv_dict[lab] = 0
            else:
                tv_dict[lab] = loggamma_mean
    else:
        raise AssertionError('True values for likelihood_name={}'.format(
            likelihood_name))
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
