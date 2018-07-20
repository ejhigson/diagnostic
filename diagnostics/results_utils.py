#!/usr/bin/env python
"""Functions for making the plots in the paper."""
import functools
import numpy as np
import scipy.integrate
import nestcheck.estimators as e
import dyPolyChord.python_likelihoods as likelihoods


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
