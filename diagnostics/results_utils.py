#!/usr/bin/env python
"""Functions for making the plots in the paper."""
import functools
import numpy as np
import scipy.integrate
import nestcheck.estimators as e
import dyPolyChord.python_likelihoods as likelihoods


R_LABEL = r'$|\theta|$'


def component_value(theta, ind=0):
    """Get parameter value from theta array given component index."""
    return theta[:, ind]


def radius(theta):
    """Function which gets the radial coordinates given a theta array in which
    each row is a position."""
    return np.sqrt(np.sum(theta ** 2, axis=1))


def default_logx_min(likelihood_name, ndim):
    """Default parameter for logx_min in diagrams."""
    return -ndim * 4


def get_ftheta_list(labels_in, ndim_max=20):
    """Get a list of ftheta functions.

    Each ftheta maps a 2d theta array to the function value for each row."""
    ftheta_dict = {R_LABEL: radius}
    for i, lab in enumerate(param_list_given_dim(ndim_max)):
        ftheta_dict[lab] = functools.partial(component_value, ind=i)
    return [ftheta_dict[lab] for lab in labels_in]


def param_latex_name(i):
    """Param latex name. Numbered starting at 1.

    N.B. the curley braces around the subscript are needed as without them
    gedist throws an error."""
    assert i > 0, i
    return r'$\theta_{{\hat{{{}}}}}$'.format(i)


def param_list_given_dim(ndim):
    """List of param names."""
    return [param_latex_name(i) for i in range(1, ndim + 1)]


def get_default_lims(like_name, ndim_max=20):
    """Get some default param limits for the likelihoods used in the paper."""
    dim_labels = param_list_given_dim(ndim_max)
    gaussian_lims = [-4, 4]  # define as used in multiple likelihoods
    lims = {}
    if like_name in ['LogGamma mix', 'LogGammaMix']:
        lims[dim_labels[0]] = [-20, 20]
        lims[dim_labels[1]] = [-20, 20]
        lims[R_LABEL] = [10, 20]
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
        lims[R_LABEL] = [0, 4]
    else:
        raise AssertionError(
            'likename={} does not have default limits'.format(like_name))
    return lims


def get_true_values_dict(prior_scale=10):
    """Numberically calculate correct values for esimators."""
    tv_fthetas = {}
    tv_fthetas[e.get_latex_name(e.param_mean)] = lambda x: x[0]
    tv_fthetas[e.get_latex_name(
        functools.partial(e.param_mean, param_ind=1))] = lambda x: x[1]
    tv_fthetas[e.get_latex_name(
        e.param_squared_mean)] = lambda x: x[0] ** 2
    tv_fthetas[e.get_latex_name(functools.partial(
        e.param_squared_mean, param_ind=1))] = lambda x: x[1] ** 2
    tv_fthetas[e.get_latex_name(
        e.r_mean)] = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
    true_values_dict = {}
    options = {"epsabs": 1.49e-11, "epsrel": 1.49e-11, 'limit': 5000}
    for like in [likelihoods.Gaussian(), likelihoods.GaussianShell(),
                 likelihoods.Rastrigin(), likelihoods.Rosenbrock()]:
        name = type(like).__name__.replace('GaussianShell', 'Gaussian shell')
        print(name)
        z = scipy.integrate.nquad(integrand_z,
                                  ranges=[(-prior_scale, prior_scale),
                                          (-prior_scale, prior_scale)],
                                  args=(like,), opts=options)
        true_values_dict[name] = {}
        true_values_dict[name][e.get_latex_name(e.logz)] = np.log(z[0])
        for ftheta_name, ftheta in tv_fthetas.items():
            val = scipy.integrate.nquad(integrand_func_not_normed,
                                        ranges=[(-prior_scale, prior_scale),
                                                (-prior_scale, prior_scale)],
                                        args=(like, ftheta), opts=options)
            true_values_dict[name][ftheta_name] = val[0] / z[0]
    return true_values_dict


def integrand_z(x2, x1, pc_like, prior_scale=10):
    return np.exp(pc_like(np.asarray([x1, x2]))[0]) / ((2 * prior_scale) ** 2)


def integrand_func_not_normed(x2, x1, pc_like, ftheta, prior_scale=10):
    return (ftheta((x1, x2)) * np.exp(pc_like(np.asarray([x1, x2]))[0])
            / ((2 * prior_scale) ** 2))
