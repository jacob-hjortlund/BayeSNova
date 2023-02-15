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
    tags += [
        os.path.split(cfg['data_cfg']['path'])[1].split(".")[0]
    ]
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
        cfg['emcee_cfg']['save_path'], "-".tags, task_name
    )
    os.makedirs(path, exist_ok=True)

    data = pd.read_csv(
        filepath_or_buffer=cfg['data_cfg']['path'], sep=cfg['data_cfg']['sep']
    )
    prep.init_global_data(data, cfg['model_cfg'])

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
            cfg['model_cfg']['independent_par_names'], cfg['model_cfg']['host_galaxy_cfg']['init_values'],
            cfg['model_cfg']['ratio_par_name']
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
            print("Resetting backend with name", cfg['emcee_cfg']['run_name'])
            backend.reset(nwalkers, ndim)
        else:
            print("Continuing from backend with name", cfg['emcee_cfg']['run_name'])

        print("\n----------------- SAMPLING ---------------------\n")
        sampler = em.EnsembleSampler(
            nwalkers, ndim, log_prob, pool=wrapped_pool.pool, backend=backend
        )
        # Set progress bar if pool is None
        use_progress_bar = wrapped_pool.pool == None
        sampler.run_mcmc(
            init_theta, cfg['emcee_cfg']['n_steps'], progress=use_progress_bar
        )
    
    t1 = time()
    total_time = t1-t0
    avg_eval_time = (total_time) / (cfg['emcee_cfg']['n_walkers'] * cfg['emcee_cfg']['n_steps'])
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
    sample_thetas = backend.get_chain(discard=burnin, thin=int(0.5*tau), flat=True)
    transposed_sample_thetas = sample_thetas.T
    sample_log_probs = backend.get_log_prob(discard=burnin, thin=int(0.5*tau), flat=True)
    idx_max = np.argmax(sample_log_probs)
    max_sample_log_prob = sample_log_probs[idx_max]
    max_thetas = sample_thetas[idx_max]
    nlp = lambda *args: -log_prob(*args)
    sol = sp_opt.minimize(nlp, max_thetas)
    
    opt_pars = sol.x
    opt_log_prob = -nlp(opt_pars)
    
    if not sol.success:
        print("Optimization not successful:", sol.message, "\n")
        print("Using init values corresponding to max sample log(P).")
        opt_log_prob = -nlp(max_thetas)
    
    print("Max sample log(P):", max_sample_log_prob)
    print("Optimized log(P):", opt_log_prob, "\n")

    # Log chain settings and log(P) values
    clearml_logger.report_single_value(name='acceptance_fraction', value=accept_frac)
    clearml_logger.report_single_value(name='tau', value=tau)
    clearml_logger.report_single_value(name='burnin', value=burnin)
    clearml_logger.report_single_value(name="max_sample_log_prob", value=max_sample_log_prob)
    clearml_logger.report_single_value(name="opt_log_prob", value=opt_log_prob)

    # Save locally in dataframe
    metrics_df = pd.DataFrame(
        {
            "acceptance_fraction": [accept_frac],
            "tau": [tau],
            "burnin": [burnin],
            "max_sample_log_prob": [max_sample_log_prob],
            "opt_log_prob": [opt_log_prob]
        }
    )
    metrics_df.to_csv(os.path.join(path, "metrics.csv"))

    print("\n-------- POSTERIOR / MARGINALIZED DIST COMPARISON -----------\n")

    if using_MPI_and_is_master or using_multiprocessing or no_pool:

        print(using_MPI_and_is_master)
        print(using_multiprocessing)
        print(no_pool)

        host_galaxy_par_names = [
            host_gal_par_name for i in range(
                prep.host_galaxy_observables.shape[1]
            )
            for host_gal_par_name in ("mu_"+str(i), "sig_"+str(i))
        ]

        par_names = (
            cfg['model_cfg']['shared_par_names'] +
            utils.gen_pop_par_names(
                cfg['model_cfg']['independent_par_names']
            ) +
            utils.gen_pop_par_names(host_galaxy_par_names) +
            [cfg['model_cfg']['ratio_par_name']]
        )

        quantiles = np.quantile(
            sample_thetas, [0.16, 0.50, 0.84], axis=0
        )
        symmetrized_stds = 0.5 * (quantiles[2] - quantiles[0])
        stds = np.std(sample_thetas, axis=0)
        map_mmap_values = np.zeros((5, len(par_names)))
        
        t2 = time()
        mmaps = np.array(list(map(utils.estimate_mmap, transposed_sample_thetas)))
        t3 = time()
        print("\nTime for MMAP estimation:", t3-t2, "s\n")
        print(mmaps)

        map_mmap_values[0] = max_thetas
        map_mmap_values[1] = mmaps
        map_mmap_values[2] = stds
        map_mmap_values[3] = symmetrized_stds
        map_mmap_values[4] = np.abs(
            map_mmap_values[0] - map_mmap_values[1]
        ) / map_mmap_values[3]

        rms_value = np.sqrt(
            np.mean(
                (map_mmap_values[0] - map_mmap_values[1])**2 / map_mmap_values[3]**2
            )
        )

        clearml_logger.report_single_value(name='rms_Z', value=rms_value)
        
        map_mmap_df = pd.DataFrame(
            map_mmap_values, index=['MAP', 'MMAP', 'sigma', 'sym_sigma', 'Z'], columns=par_names
        )
        clearml_logger.report_table(
            title='MAP_MMAP_Distance',
            series='MAP_MMAP_Distance',
            iteration=0,
            table_plot=map_mmap_df
        )

        map_mmap_df.to_csv(os.path.join(path, "map_mmap.csv"))
        print(f"\nRMS value: {rms_value}\n")

    print("\n-------- PARAM VALUES / ERRORS / Z-SCORES -----------\n")

    val_errors = np.concatenate(
        (quantiles, symmetrized_stds[None, :]),
        axis=0
    )
    val_errors_df = pd.DataFrame(
        val_errors, index=['MMAP', '16th', '50th', '84th', 'sym_sigma'], columns=par_names
    )

    max_symmetric_error = np.max(
        symmetrized_stds[
            len(cfg['model_cfg']['shared_par_names']):-1
        ].reshape(-1, 2), axis=-1
    )
    param_diff = np.squeeze(
        np.diff(
            quantiles[1][len(cfg['model_cfg']['shared_par_names']):-1].reshape(-1, 2),
            axis=1
        )
    )
    param_z_scores = param_diff / max_symmetric_error

    z_score_df = pd.DataFrame(
        param_z_scores, index=['Z'], columns=(
            cfg['model_cfg']['independent_par_names'] + host_galaxy_par_names
        )
    )

    clearml_logger.report_table(
        title='Medians_and_Errors',
        series='Medians_and_Errors',
        iteration=0,
        table_plot=val_errors_df
    )

    clearml_logger.report_table(
        title='Z_Scores',
        series='Z_Scores',
        iteration=0,
        table_plot=z_score_df
    )

    val_errors_df.to_csv(os.path.join(path, "medians_and_errors.csv"))
    z_score_df.to_csv(os.path.join(path, "z_scores.csv"))

    print("\n----------------- PLOTS ---------------------\n")
    n_shared = len(cfg['model_cfg']['shared_par_names'])
    n_independent = len(cfg['model_cfg']['independent_par_names'])
    labels = (
        cfg['model_cfg']['shared_par_names'] +
        cfg['model_cfg']['independent_par_names'] +
        host_galaxy_par_names +
        [cfg['model_cfg']['ratio_par_name']]
    )

    fx = sample_thetas[:, :n_shared]
    fig_pop_2 = None

    # Handling in case of independent parameters
    if n_independent > 0:
        fx = np.concatenate(
            (
                fx, sample_thetas[:, n_shared:-1:2]
            ), axis=-1
        )

        sx_list = [
            sample_thetas[:, :n_shared],
            sample_thetas[:, n_shared+1:-1:2],
            sample_thetas[:,-1][:, None]
        ]
        sx = np.concatenate(sx_list, axis=-1)

        fig_pop_2 = corner.corner(data=sx, color='r', **cfg['plot_cfg'])
    
    fx_list = [fx, sample_thetas[:,-1][:, None]]
    fx = np.concatenate(fx_list, axis=-1)

    fig_pop_1 = corner.corner(data=fx, fig=fig_pop_2, labels=labels, **cfg['plot_cfg'])
    fig_pop_1.tight_layout()
    fig_pop_1.suptitle('Corner plot', fontsize=int(2 * cfg['plot_cfg']['label_kwargs']['fontsize']))
    fig_pop_1.savefig(
        os.path.join(path, cfg['emcee_cfg']['run_name']+"_corner.pdf")
    )

    full_chain = backend.get_chain()
    fig, axes = plt.subplots(
        ndim, figsize=(10, int(np.ceil(7*ndim/3))), sharex=True
    )
    for i in range(ndim):
        ax = axes[i]
        ax.plot(full_chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(full_chain))
        ax.set_ylabel(par_names[i], fontsize=cfg['plot_cfg']['label_kwargs']['fontsize'])
        ax.axvline(burnin, color='r', alpha=0.5)
    axes[-1].set_xlabel("step number", fontsize=cfg['plot_cfg']['label_kwargs']['fontsize'])
    fig.tight_layout()
    if cfg['model_cfg']['use_sigmoid']:
        suffix = "_transformed"
    else:
        suffix = ""
    fig.subplots_adjust(top=0.1)
    fig.suptitle("Walkers" + suffix, fontsize=int(2 * cfg['plot_cfg']['label_kwargs']['fontsize']))
    fig.savefig(
        os.path.join(path, cfg['emcee_cfg']['run_name']+suffix+"_walkers.pdf")
    )

if __name__ == "__main__":
    main()