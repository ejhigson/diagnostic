#!/usr/bin/env python
"""Functions for load PolyChord data made with generate_data.py."""
import os
import pandas as pd
import nestcheck.ns_run_utils
import nestcheck.plots
import nestcheck.data_processing
import nestcheck.parallel_utils
import nestcheck.diagnostics_tables
import nestcheck.estimators as e
import dyPolyChord.output_processing
import diagnostics.results_utils
import diagnostics.settings


def get_file_root(likelihood_name, ndim, nlive, nrepeats, **kwargs):
    """File root with default prior and ndim"""
    prior_name = kwargs.pop('prior_name', 'Uniform')
    prior_scale = kwargs.pop('prior_scale', 30)
    return dyPolyChord.output_processing.settings_root(
        likelihood_name.title().replace(' ', '').replace('gamma', 'Gamma'),
        prior_name, ndim, prior_scale=prior_scale, nlive_const=nlive,
        nrepeats=nrepeats, dynamic_goal=None)


def get_run_list(likelihood_name, nrun, **kwargs):
    """Helper function for loading lists of nested sampling runs."""
    ndim, nlive, nrepeats = diagnostics.settings.get_default_nd_nl_nr()
    ndim = kwargs.pop('ndim', ndim)
    nlive = kwargs.pop('nlive', nlive)
    nrepeats = kwargs.pop('nrepeats', nrepeats)
    nrun_start = kwargs.pop('nrun_start', 1)
    file_root = get_file_root(likelihood_name, ndim, nlive, nrepeats, **kwargs)
    files = [file_root + '_' + str(i).zfill(3) for i in
             range(nrun_start, nrun + nrun_start)]
    return nestcheck.data_processing.batch_process_data(files)


def get_run_list_dict(likelihood_list, nrun, **kwargs):
    """Wrapper for getting dict with a run list for each likelihood."""
    run_list_dict = {}
    for likelihood_name in likelihood_list:
        run_list_dict[likelihood_name] = get_run_list(
            likelihood_name, nrun, **kwargs)
    return run_list_dict


def get_results_df(likelihood_list, nd_nl_nr_list, **kwargs):
    """Get a big pandas multiindex data frame with results for different
    likelihoods, nlives and nrepeats."""
    estimator_list = kwargs.pop(
        'estimator_list', diagnostics.settings.get_default_estimator_list())
    estimator_names = kwargs.pop(
        'estimator_names',
        [e.get_latex_name(est) for est in estimator_list])
    assert len(estimator_list) == len(estimator_names)
    n_simulate = kwargs.pop('n_simulate', 100)
    nrun = kwargs.pop('nrun', 100)
    summary = kwargs.pop('summary', True)
    include_rmse = kwargs.pop('include_rmse', False)
    include_true_values = kwargs.pop('include_true_values', False)
    results_list = []
    progress = nestcheck.parallel_utils.select_tqdm()
    for likelihood_name in progress(likelihood_list, leave=False,
                                    desc='likelihoods'):
        for ndim, nlive, nrepeats in progress(
                nd_nl_nr_list, leave=False, desc='ndim, nlive, nrepeats'):
            # Get the cache save name
            file_root = get_file_root(likelihood_name, ndim, nlive, nrepeats)
            save_name = 'cache/errors_df_{}_{}runs_{}sim'.format(
                file_root, nrun, n_simulate)
            if kwargs.get('thread_pvalue', False):
                save_name += '_td'
            if kwargs.get('bs_stat_dist', False):
                save_name += '_bd'
            # Get results
            # If cache exists it will be loaded without checking for ns runs
            # If neither cache nor runs exist, will continue to next loop
            try:
                if os.path.exists(save_name + '.pkl'):
                    run_list = None
                else:
                    print('File not found: {}.pkl'.format(save_name))
                    run_list = get_run_list(likelihood_name, nrun, nlive=nlive,
                                            nrepeats=nrepeats, ndim=ndim)
                df = nestcheck.diagnostics_tables.run_list_error_values(
                    run_list, estimator_list, estimator_names, n_simulate,
                    save_name=save_name, **kwargs)
                if summary:
                    true_values = diagnostics.results_utils.get_true_values(
                        likelihood_name, ndim, estimator_names)
                    df = nestcheck.diagnostics_tables.error_values_summary(
                        df, true_values=true_values, include_rmse=include_rmse,
                        include_true_values=include_true_values)
                new_inds = ['likelihood', 'ndim', 'nlive', 'nrepeats']
                df['likelihood'] = likelihood_name
                df['ndim'] = ndim
                df['nlive'] = nlive
                df['nrepeats'] = nrepeats
                order = new_inds + list(df.index.names)
                df.set_index(new_inds, drop=True, append=True, inplace=True)
                df = df.reorder_levels(order)
                results_list.append(df)
            except OSError:
                pass
    results = pd.concat(results_list)
    return results
