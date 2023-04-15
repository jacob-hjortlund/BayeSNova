import hydra
import omegaconf
import os
import corner

import emcee as em
import numpy as np
import pandas as pd
import clearml as cl
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt
import src.utils as utils
import src.model as model
import src.preprocessing as prep
import src.postprocessing as post

from time import time
from mpi4py import MPI

@hydra.main(
    version_base=None, config_path="configs", config_name="config"
)
def main(cfg: omegaconf.DictConfig) -> None:   

    # Setup clearml
    task_name = utils.create_task_name(cfg)
    if len(cfg['model_cfg']['independent_par_names']) > 0:
        tags = ["-".join(cfg['model_cfg']['independent_par_names'])]
    else:
        tags = ["all_shared"]
    if len(cfg['model_cfg']['cosmology_par_names']) > 0:
        tags += ["-".join(cfg['model_cfg']['cosmology_par_names'])]
    tags += [
        os.path.split(cfg['data_cfg']['train_path'])[1].split(".")[0]
    ]
    if cfg['model_cfg']['host_galaxy_cfg']['use_properties']:
        tags += cfg['model_cfg']['host_galaxy_cfg']['property_names']
    tags += cfg['clearml_cfg']['tags']

    using_MPI_and_is_master = cfg['emcee_cfg']['pool_type'] == 'MPI' and MPI.COMM_WORLD.rank == 0
    using_multiprocessing = cfg['emcee_cfg']['pool_type'] == 'MP'
    no_pool = cfg['emcee_cfg']['pool_type'] == ''

    if using_MPI_and_is_master or using_multiprocessing or no_pool:
        cl.Task.set_offline(offline_mode=cfg['clearml_cfg']['offline_mode'])
        task = cl.Task.init(
            project_name=cfg['clearml_cfg']['project_name'],
            task_name=task_name, tags=tags, task_type=cfg['clearml_cfg']['task_type']
        )
        clearml_logger = task.get_logger()

    # Setup results dir
    path = os.path.join(
        cfg['emcee_cfg']['save_path'], "-".join(tags), task_name
    )
    os.makedirs(path, exist_ok=True)

    data = pd.read_csv(
        filepath_or_buffer=cfg['data_cfg']['train_path'], sep=cfg['data_cfg']['sep']
    )
    
    if cfg['data_cfg']['volumetric_rates_path'] and cfg['model_cfg']['use_volumetric_rates']:
        volumetric_rates = pd.read_csv(
            filepath_or_buffer=cfg['data_cfg']['volumetric_rates_path'], sep=cfg['data_cfg']['sep']
        )
    else:
        volumetric_rates = None

    prep.init_global_data(data, volumetric_rates, cfg['model_cfg'])

    log_prob = model.Model()

    t0 = time()
    with utils.PoolWrapper(cfg['emcee_cfg']['pool_type']) as wrapped_pool:
        
        if wrapped_pool.is_mpi:
            wrapped_pool.check_if_master()

        # Print cfg
        print("\n----------------- CONFIG ---------------------\n")
        print(omegaconf.OmegaConf.to_yaml(cfg),"\n")

        print("\n----------------- SETUP ---------------------\n")
        init_theta = utils.prior_initialisation(
            cfg['model_cfg']['prior_bounds'], cfg['model_cfg']['init_values'], cfg['model_cfg']['shared_par_names'],
            cfg['model_cfg']['independent_par_names'], cfg['model_cfg']['ratio_par_name'], cfg['model_cfg']['cosmology_par_names'],
            cfg['model_cfg']['use_physical_ratio'], cfg['model_cfg']['host_galaxy_cfg']['use_properties'],
            cfg['model_cfg']['host_galaxy_cfg']['property_names'], cfg['model_cfg']['host_galaxy_cfg']['init_values'], 
        )
        init_theta = init_theta + 3e-2 * np.random.rand(cfg['emcee_cfg']['n_walkers'], len(init_theta))
        _ = log_prob(init_theta[0]) # call func once befor loop to jit compile
        nwalkers, ndim = init_theta.shape

        # Setup emcee backend
        filename = os.path.join(
            path, cfg['emcee_cfg']['file_name']
        )
        backend = em.backends.HDFBackend(filename, name=cfg['emcee_cfg']['run_name'])
        if not cfg['emcee_cfg']['continue_from_chain']:
            n_steps = cfg['emcee_cfg']['n_steps']
            print(
                "Resetting backend with name", cfg['emcee_cfg']['run_name']
            )
            backend.reset(nwalkers, ndim)
        else:
            n_steps = cfg['emcee_cfg']['n_steps'] - backend.iteration
            print("Continuing from backend with name", cfg['emcee_cfg']['run_name'])
            print(f"Running for {n_steps}")

        print("\n----------------- SAMPLING ---------------------\n")
        sampler = em.EnsembleSampler(
            nwalkers, ndim, log_prob, pool=wrapped_pool.pool, backend=backend
        )
        # Set progress bar if pool is None
        use_progress_bar = wrapped_pool.pool == None
        sampler.run_mcmc(
            init_theta, n_steps, progress=use_progress_bar
        )
    
    t1 = time()
    total_time = t1-t0
    avg_eval_time = (total_time) / (cfg['emcee_cfg']['n_walkers'] * n_steps)
    print(f"\nTotal MCMC time:", total_time)
    print(f"Avg. time pr. step: {avg_eval_time} s\n")

    # If using sigmoid, transform samples
    if cfg['model_cfg']['use_sigmoid']:
        backend = utils.transformed_backend(
            backend, filename, name=cfg['emcee_cfg']['run_name']+"_transformed",
            sigmoid_cfg=cfg['model_cfg']['sigmoid_cfg'], shared_par_names=cfg['model_cfg']['shared_par_names']
        )
    accept_frac = np.mean(sampler.acceptance_fraction)
    print("\nMean accepance fraction:", accept_frac, "\n")

    try:
        taus = backend.get_autocorr_time(
            tol=cfg['emcee_cfg']['tau_tol']
        )
        tau = np.max(taus)
        print("\nAtuocorr lengths:\n", taus)
        print("Max autocorr length:", tau, "\n")
    except Exception as e:
        print("\nGot exception:\n")
        print(e, "\n")
        tau = cfg['emcee_cfg']['default_tau']
        print("\nUsing default tau:", tau, "\n")

    burnin = np.max(
        [int(5 * tau), int(cfg['emcee_cfg']['default_burnin'])]
    )

    print(f"\nThinning by {burnin} steps\n")

    print("\n----------------- MAX LOG(P) ---------------------\n")
    # TODO: Redo without interpolation
    
    sample_thetas, max_thetas = post.mcmc_statistics(
        log_prob, backend, tau, burnin, accept_frac, path, clearml_logger
    )

    print("\n-------- POSTERIOR / MARGINALIZED DIST COMPARISON -----------\n")

    par_names, host_galaxy_par_names = post.get_par_names(cfg)
    if using_MPI_and_is_master or using_multiprocessing or no_pool:

        quantiles, symmetrized_stds = post.map_mmap_comparison(
            sample_thetas, max_thetas, par_names, path, clearml_logger
        )

    print("\n-------- PARAM VALUES / ERRORS / Z-SCORES -----------\n")

    post.parameter_values(
        quantiles, symmetrized_stds, par_names,
        host_galaxy_par_names, cfg, path, clearml_logger
    )

    print("\n----------------- PLOTS ---------------------\n")
    labels = (
        cfg['model_cfg']['shared_par_names'] +
        cfg['model_cfg']['independent_par_names'] +
        host_galaxy_par_names +
        cfg['model_cfg']['cosmology_par_names'] +
        (not cfg['model_cfg']['use_physical_ratio']) * [cfg['model_cfg']['ratio_par_name']]
    )

    (
        full_membership_quantiles, sn_membership_quantiles, host_membership_quantiles
    ) = post.get_membership_quantiles(
        backend, burnin, tau
    )

    cm, cm_norm_full, mapper_full, pop1_color, pop2_color = post.setup_colormap(
        full_membership_quantiles, sn_membership_quantiles,
        host_membership_quantiles, cfg
    )

    post.corner_plot(
        sample_thetas, pop1_color, pop2_color, labels, cfg, path
    )

    post.chain_plot(
        backend, burnin, par_names, cfg, path
    )

    quantiles_list = [
        full_membership_quantiles, sn_membership_quantiles,
    ]
    titles_list = ["Full", "SN",]
    if cfg['model_cfg']['host_galaxy_cfg']['use_properties']:
        quantiles_list.append(host_membership_quantiles)
        titles_list.append("Host")
    
    post.membership_histogram(
        quantiles_list, titles_list, mapper_full,
        prep.idx_reordered_calibrator_sn, cfg, path
    )

    available_properties_names = data.keys()[::2]
    for property_name in available_properties_names:

        host_property = data[property_name].to_numpy()
        host_property_errors = data[property_name+"_err"].to_numpy()
        if np.all(host_property == prep.NULL_VALUE):
            continue

        print("\nPlotting property: ", property_name)
        
        post.observed_property_vs_membership(
            property_name, host_property, host_property_errors,
            full_membership_quantiles, cm, cm_norm_full, mapper_full,
            prep.idx_unique_sn, prep.idx_duplicate_sn,
            prep.idx_reordered_calibrator_sn, cfg, path
        )


if __name__ == "__main__":
    main()