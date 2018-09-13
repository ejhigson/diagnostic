#!/usr/bin/env python
"""Generate data using PolyChord.

Runs the PolyChord data used in the paper and stores it in the directory
'chains'.

Requires:
    * diagnostic module (and its dependencies),
    * PolyChord >= v1.14

Results are generated using dyPolyChord's interface for convenience, but
all the results in the paper use standard (rather than dynamic) nested
sampling.

### Random seeding

Random seeding is used for reproducible results, which is only
possible when PolyChord is run *without* MPI due to the unpredictable
order in which threads will provide samples (see the PolyChord
documentation for more details). As generating repeated runs is "embarrassingly
parallel" we can instead parallelise using concurrent.futures via nestcheck's
parallel_apply function.

Note also that PolyChord's random number generator can vary between systems and
compilers, so your results may not exactly match those in the paper.
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
import diagnostic.results_utils
import diagnostic.data_loading
import diagnostic.settings
try:
    # This initialises MPI, allowing running multiple runs from the same python
    # instance even if PolyChord is installed with MPI (so you don't have to
    # reinstall it without MPI).
    from mpi4py import MPI  # pylint: disable=unused-import
except ImportError:
    pass


def main():
    """Generate PolyChord runs. Also processes the results into a DataFrame and
    caches it.

    Nested sampling runs are generating for different settings by looping over:

        * likelihood_list: different likelihoods;
        * nd_nl_nr_list: list of tuples, each containing (number of dimensions,
          nlive, nrepeats);
        * inds: labels for repeated runs to generate with each setting.
    """
    # Settings
    # --------
    # If true, many runs are made at the same time via concurrent.futures
    parallel = True
    # Run settings
    inds = list(range(1, 101))
    # dimensions, nlive, nrepeat settings
    # -----------------------------------
    nd_nl_nr_list = diagnostic.settings.get_nd_nl_nr_list()
    # Likelihood and prior settings
    # -----------------------------
    likelihood_list = [likelihoods.LogGammaMix(),
                       likelihoods.Gaussian(sigma=1)]
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
    if 'ed' in os.getcwd().split('/'):
        # running on laptop - don't use all the processors so I can do other
        # stuff without everything getting slow
        max_workers = 6
    else:
        max_workers = None  # running on cluster
    print('Running with max_workers={}'.format(max_workers))
    prior_scale = 30
    prior = priors.Uniform(-prior_scale, prior_scale)
    # Before running in parallel make sure base_dir exists, as if multiple
    # threads try to make one at the same time mkdir throws an error.
    if not os.path.exists(settings_dict['base_dir']):
        os.makedirs(settings_dict['base_dir'])
    if not os.path.exists(settings_dict['base_dir'] + '/clusters'):
        os.makedirs(settings_dict['base_dir'] + '/clusters')
    for likelihood in likelihood_list:
        for ndim, nlive, num_repeats in nd_nl_nr_list:
            run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
                likelihood, prior, ndim)
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
            desc = '{} ndim={} nlive={} nrep={}'.format(
                type(likelihood).__name__, ndim, nlive, num_repeats)
            nestcheck.parallel_utils.parallel_apply(
                run_func, settings_list,
                max_workers=max_workers, parallel=parallel,
                tqdm_kwargs={'desc': desc, 'leave': True})
            # Cache results DataFrame
            # -----------------------
            diagnostic.data_loading.get_results_df(
                [type(likelihood).__name__.replace('Mix', ' mix')],
                [(ndim, nlive, num_repeats)], n_simulate=100,
                nrun=inds[-1], summary=True, save=True, load=True,
                thread_pvalue=False, bs_stat_dist=False,
                include_rmse=True, include_true_values=True, parallel=True)
            if ((ndim, nlive, num_repeats) ==
                    diagnostic.settings.get_default_nd_nl_nr()):
                # Cache bs stat and thread values df too
                # Use summary=True as caching is done to raw values and this
                # protects against unexpected kwarg errors as summary pops some
                # kwargs before getting values
                diagnostic.data_loading.get_results_df(
                    [type(likelihood).__name__.replace('Mix', ' mix')],
                    [(ndim, nlive, num_repeats)], n_simulate=100,
                    nrun=inds[-1], summary=True, save=True, load=True,
                    thread_pvalue=True, bs_stat_dist=True,
                    include_rmse=True, include_true_values=True, parallel=True)


if __name__ == '__main__':
    main()
