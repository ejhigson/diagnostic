#!/usr/bin/env python
"""
# Generate data using PolyChord
# -----------------------------

Runs the PolyChord data used in the paper and stores it in the directory
'chains'.

Requies:
    * PolyChord (install without MPI)
    * nestcheck
    * dyPolyChord

dyPolyChord's interface for convenience, but for simplicity all the results
in the paper use standard (rather than dynamic) nested sampling.

### Random seeding

Random seeding is used for reproducible results, which is only
possible when PolyChord is installed *without* MPI due to the unpredictable
order in which threads will provide samples (see the PolyChord
documentation for more details). As generating repeated runs is embarrassingly
parallel we can instead parallelise using concurrent.futures via nestcheck's
parallel_apply function.

Note also that PolyChord's random number generator can vary between systems and
compilers, so your results may not exactly match those in the paper (which were
run on Ubuntu 18.04 with PolyChord compiled using gfortran 7.3.0).

These results can be run with a python or C++ likelihood by setting the
'compiled' variable to True or False. C++ is *much* faster but requires
compiling with PolyChord.

### Compiling the C++ likelihood

With PolyChord 1.14, C++ likelihoods like gaussian.cpp can be compiled with
the following commands. This assumes PolyChord is already installed, without
MPI, in location path_to_pc/PolyChord.

    $ cp gaussian.cpp
    ~/path_to_pc/PolyChord/likelihoods/CC_ini/CC_ini_likelihood.cpp
    $ cd ~/path_to_pc/PolyChord
    $ make polychord_CC_ini

This produces an executable at PolyChord/bin/polychord_CC_ini which you can
move back to the current directory to run (or edit RunCompiledPolyChord's
executable path accordingly).

For more details see PolyChord's README.
"""
import copy
import os
import nestcheck.parallel_utils
import dyPolyChord.python_likelihoods as likelihoods
import dyPolyChord.python_priors as priors
import dyPolyChord.output_processing
import dyPolyChord.polychord_utils
import dyPolyChord.pypolychord_utils
import dyPolyChord


def main():
    """Generate PolyChord runs."""
    # Settings
    # --------
    # Run settings
    inds = list(range(1, 11))
    parallel = True
    max_workers = 6
    compiled = False
    # nlive and nrepeat settings
    nlive_nrepeats_list = []
    nr_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    nl_list = [10, 20, 50, 200, 500, 1000]
    for nr in nr_list:
        nlive_nrepeats_list.append((100, nr))
    for nl in nl_list:
        nlive_nrepeats_list.append((nl, 5))
    nlive_nrepeats_list = [(100, 5)]
    # Likelihood and prior settings
    # -----------------------------
    ndim = 2
    prior_scale = 10
    likelihood_list = [likelihoods.Gaussian(), likelihoods.GaussianShell(),
                       likelihoods.Rastrigin(), likelihoods.Rosenbrock()]
    # PolyChord settings
    settings_dict = {
        'do_clustering': True,
        'posteriors': False,
        'equals': False,
        'base_dir': 'chains',
        'feedback': -1,
        'precision_criterion': 0.001,
        'nlives': {},
        'write_dead': True,
        'write_stats': True,
        'write_paramnames': False,
        'write_prior': False,
        'write_live': False,
        'write_resume': False,
        'read_resume': False,
        'max_ndead': -1,
        'cluster_posteriors': False,
        'boost_posterior': 0.0}
    prior = priors.Uniform(-prior_scale, prior_scale)
    # Before running in parallel make sure base_dir exists, as if multiple
    # threads try to make one at the same time mkdir throws an error.
    if not os.path.exists(settings_dict['base_dir']):
        os.makedirs(settings_dict['base_dir'])
    if not os.path.exists(settings_dict['base_dir'] + '/clusters'):
        os.makedirs(settings_dict['base_dir'] + '/clusters')
    for likelihood in likelihood_list:
        if not compiled:
            run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
                likelihood, prior, ndim)
        else:
            assert type(prior).__name__ == 'Uniform', (
                'Prior={} - you may need to change get_prior_block_str '
                'arguments'.format(type(prior).__name__))
            assert len(likelihood_list) == 1
            prior_str = dyPolyChord.polychord_utils.get_prior_block_str(
                'uniform', [float(-prior_scale), float(prior_scale)], ndim)
            run_func = dyPolyChord.polychord_utils.RunCompiledPolyChord(
                './polychord_CC_ini', prior_str)
        for nlive, num_repeats in nlive_nrepeats_list:
            # make list of settings dictionaries for the different repeats
            file_root = dyPolyChord.output_processing.settings_root(
                type(likelihood).__name__,
                type(prior).__name__, ndim,
                prior_scale=prior_scale, nrepeats=num_repeats,
                nlive_const=nlive, dynamic_goal=None)
            settings_dict['nlive'] = nlive
            settings_dict['num_repeats'] = num_repeats
            settings_list = []
            for extra_root in inds:
                settings = copy.deepcopy(settings_dict)
                settings['seed'] = extra_root
                settings['file_root'] = file_root
                settings['file_root'] += '_' + str(extra_root).zfill(3)
                settings_list.append(settings)
            # Do the nested sampling
            # ----------------------
            # For standard nested sampling just run PolyChord
            desc = '{} nlive={} nrep={}'.format(
                type(likelihood).__name__, nlive, num_repeats)
            nestcheck.parallel_utils.parallel_apply(
                run_func, settings_list,
                max_workers=max_workers, parallel=parallel,
                tqdm_kwargs={'desc': desc, 'leave': True})


if __name__ == '__main__':
    main()
