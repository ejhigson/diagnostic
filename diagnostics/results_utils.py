#!/usr/bin/env python
"""Utilites for running results."""
# import functools
# import copy
import tqdm
import pandas as pd
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.gridspec
# import mpl_toolkits
import nestcheck.ns_run_utils
import nestcheck.plots
import nestcheck.data_processing
# import nestcheck.plots
import nestcheck.diagnostics_tables
import nestcheck.estimators as e
import dyPolyChord.output_processing


def get_results_df(likelihood_list, nlive_nrepeats_list, estimator_list,
                   **kwargs):
    """Get a big pandas multiindex data frame with results for different
    likelihoods, nlives and nrepeats."""
    estimator_names = kwargs.pop(
        'estimator_names',
        [e.get_latex_name(est) for est in estimator_list])
    n_simulate = kwargs.pop('n_simulate', 5)
    n_runs = kwargs.pop('n_runs', 100)
    ndim = kwargs.pop('ndim', 2)
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
            file_root = dyPolyChord.output_processing.settings_root(
                likelihood_name, 'uniform', ndim, nlive_const=nlive,
                nrepeats=nrepeats, dynamic_goal=None, prior_scale=10)
            run_list = nestcheck.data_processing.batch_process_data(
                [file_root + '_' + str(i) for i in range(1, n_runs + 1)],
                func_kwargs={'errors_to_handle': ()})
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
