#!/usr/bin/env python
"""Utilites for loading data from generate_data.py."""
import tqdm
import pandas as pd
import numpy as np
import nestcheck.ns_run_utils
import nestcheck.plots
import nestcheck.data_processing
import nestcheck.diagnostics_tables
import nestcheck.estimators as e
import dyPolyChord.output_processing


def get_file_root(likelihood_name, nlive, nrepeats, **kwargs):
    """File root with default prior and ndim"""
    prior_name = kwargs.pop('prior_name', 'Uniform')
    prior_scale = kwargs.pop('prior_scale', 10)
    ndim = kwargs.pop('ndim', 2)
    return dyPolyChord.output_processing.settings_root(
        likelihood_name.title().replace(' ', ''), prior_name, ndim,
        prior_scale=prior_scale, nlive_const=nlive, nrepeats=nrepeats,
        dynamic_goal=None)


def get_run_list(likelihood_name, nrun, **kwargs):
    """Helper function for loading lists of nested sampling runs."""
    nlive = kwargs.pop('nlive', 100)
    nrepeats = kwargs.pop('nrepeats', 5)
    nrun_start = kwargs.pop('nrun_start', 1)
    file_root = get_file_root(likelihood_name, nlive, nrepeats, **kwargs)
    files = [file_root + '_' + str(i).zfill(3) for i in
             range(nrun_start, nrun + nrun_start)]
    return nestcheck.data_processing.batch_process_data(files, **kwargs)


def get_run_list_dict(likelihood_list, nrun, **kwargs):
    """Wrapper for getting dict with a run list for each likelihood."""
    run_list_dict = {}
    for likelihood_name in likelihood_list:
        run_list_dict[likelihood_name] = get_run_list(
            likelihood_name, nrun, **kwargs)
    return run_list_dict


def get_results_df(likelihood_list, nlive_nrepeats_list, estimator_list,
                   **kwargs):
    """Get a big pandas multiindex data frame with results for different
    likelihoods, nlives and nrepeats."""
    estimator_names = kwargs.pop(
        'estimator_names',
        [e.get_latex_name(est) for est in estimator_list])
    n_simulate = kwargs.pop('n_simulate', 5)
    nrun = kwargs.pop('nrun', 100)
    summary = kwargs.pop('summary', True)
    true_values_dict = kwargs.pop('true_values_dict', None)
    results_list = []
    assert len(estimator_list) == len(estimator_names)
    for likelihood_name in tqdm.tqdm_notebook(likelihood_list, leave=False,
                                              desc='likelihoods'):
        if true_values_dict is not None:
            true_values = np.full(len(estimator_names), np.nan)
            for i, name in enumerate(estimator_names):
                try:
                    true_values[i] = true_values_dict[likelihood_name][name]
                except KeyError:
                    pass
        else:
            true_values = None
        for nlive, nrepeats in tqdm.tqdm_notebook(
                nlive_nrepeats_list, leave=False, desc='nlive_nrepeats'):
            run_list = get_run_list(likelihood_name, nrun, nlive=nlive,
                                    nrepeats=nrepeats)
            file_root = get_file_root(likelihood_name, nlive, nrepeats)
            save_name = 'cache/errors_df_{}_{}runs_{}sim'.format(
                file_root, len(run_list), n_simulate)
            if kwargs.get('thread_pvalue', False):
                save_name += '_td'
            if kwargs.get('bs_stat_dist', False):
                save_name += '_bd'
            if summary:
                df_temp = nestcheck.diagnostics_tables.run_list_error_summary(
                    run_list, estimator_list, estimator_names, n_simulate,
                    save_name=save_name, true_values=true_values, **kwargs)
            else:
                df_temp = nestcheck.diagnostics_tables.run_list_error_values(
                    run_list, estimator_list, estimator_names, n_simulate,
                    save_name=save_name, **kwargs)
            new_inds = ['likelihood', 'nlive', 'nrepeats']
            df_temp['likelihood'] = likelihood_name
            df_temp['nlive'] = nlive
            df_temp['nrepeats'] = nrepeats
            order = new_inds + list(df_temp.index.names)
            df_temp.set_index(new_inds, drop=True, append=True, inplace=True)
            df_temp = df_temp.reorder_levels(order)
            results_list.append(df_temp)
    results = pd.concat(results_list)
    return results
