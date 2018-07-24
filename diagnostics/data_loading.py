#!/usr/bin/env python
"""Functions for load PolyChord data made with generate_data.py."""
import warnings
import tqdm
import pandas as pd
import nestcheck.ns_run_utils
import nestcheck.plots
import nestcheck.data_processing
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
    n_simulate = kwargs.pop('n_simulate', 100)
    nrun = kwargs.pop('nrun', 100)
    summary = kwargs.pop('summary', True)
    results_list = []
    assert len(estimator_list) == len(estimator_names)
    for likelihood_name in tqdm.tqdm_notebook(likelihood_list, leave=False,
                                              desc='likelihoods'):
        for ndim, nlive, nrepeats in tqdm.tqdm_notebook(
                nd_nl_nr_list, leave=False, desc='ndim, nlive, nrepeats'):
            run_list = get_run_list(likelihood_name, nrun, nlive=nlive,
                                    nrepeats=nrepeats, ndim=ndim)
            file_root = get_file_root(likelihood_name, ndim, nlive, nrepeats)
            save_name = 'cache/errors_df_{}_{}runs_{}sim'.format(
                file_root, len(run_list), n_simulate)
            if kwargs.get('thread_pvalue', False):
                save_name += '_td'
            if kwargs.get('bs_stat_dist', False):
                save_name += '_bd'
            true_values = diagnostics.results_utils.get_true_values(
                likelihood_name, ndim, estimator_names)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                if summary:
                    df = nestcheck.diagnostics_tables.run_list_error_summary(
                        run_list, estimator_list, estimator_names, n_simulate,
                        save_name=save_name, true_values=true_values, **kwargs)
                else:
                    df = nestcheck.diagnostics_tables.run_list_error_values(
                        run_list, estimator_list, estimator_names, n_simulate,
                        save_name=save_name, **kwargs)
            new_inds = ['likelihood', 'ndim', 'nlive', 'nrepeats']
            df['likelihood'] = likelihood_name
            df['ndim'] = ndim
            df['nlive'] = nlive
            df['nrepeats'] = nrepeats
            order = new_inds + list(df.index.names)
            df.set_index(new_inds, drop=True, append=True, inplace=True)
            df = df.reorder_levels(order)
            results_list.append(df)
    results = pd.concat(results_list)
    return results
