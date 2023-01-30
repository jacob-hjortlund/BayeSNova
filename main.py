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
import src.preprocessing as prep
import src.probabilities as prob

from time import time

@hydra.main(
    version_base=None, config_path="configs", config_name="config"
)
def main(cfg: omegaconf.DictConfig) -> None:   

    # Setup clearml
    cl.Task.set_offline(offline_mode=cfg['clearml_cfg']['offline_mode'])
    task_name = utils.create_task_name(cfg)
    tags = ["-".join(cfg['model_cfg']['independent_par_names'])] + cfg['clearml_cfg']['tags']
    task = cl.Task.init(
        project_name=cfg['clearml_cfg']['project_name'],
        task_name=task_name, tags=tags, task_type=cfg['clearml_cfg']['task_type']
    )
    clearml_logger = task.get_logger()

    # Setup results dir
    path = os.path.join(
        cfg['emcee_cfg']['save_path'], tags[0], task_name
    )
    os.makedirs(path, exist_ok=True)

    # Import data
    data = pd.read_csv(
        filepath_or_buffer=cfg['data_cfg']['path'], sep=cfg['data_cfg']['sep']
    )

    # Preprocess
    sn_covs = prep.build_covariance_matrix(data.to_numpy())
    sn_mb = data['mB'].to_numpy()
    sn_s = data['x1'].to_numpy()
    sn_c = data['c'].to_numpy()
    sn_z = data['z'].to_numpy()

    # Gen log_prob function
    log_prob = prob.generate_log_prob(
        cfg['model_cfg'], sn_covs=sn_covs,
        sn_mb=sn_mb, sn_s=sn_s, sn_c=sn_c, sn_z=sn_z
    )

    t0 = time()
    with utils.PoolWrapper(cfg['emcee_cfg']['pool_type']) as wrapped_pool:
        
        if wrapped_pool.is_mpi:
            wrapped_pool.check_if_master()

        # Print cfg
        print("\n----------------- CONFIG ---------------------\n")
        print(omegaconf.OmegaConf.to_yaml(cfg),"\n")

        print("\n----------------- SETUP ---------------------\n")
        # Setup init pos and log_prob
        theta = np.array([
                -0.14, 3.1, 3.7, 0.6, 2.9, -19.3, -19.3, -0.09, -0.09, 0.05, 0.03, -0.25, 0.1, 1.1, 1.1, 0.04, 0.04, 0.4
            ]) 
        #init_theta = theta + 3e-2 * np.random.rand(cfg['emcee_cfg']['n_walkers'], len(theta))
        init_theta = utils.prior_initialisation(
            cfg['model_cfg']['prior_bounds'], cfg['model_cfg']['init_values'], cfg['model_cfg']['shared_par_names'],
            cfg['model_cfg']['independent_par_names'], cfg['model_cfg']['ratio_par_name']
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

    print("\nMean accepance fraction:", np.mean(sampler.acceptance_fraction), "\n")

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

    print("\n----------------- PLOTS ---------------------\n")
    n_shared = len(cfg['model_cfg']['shared_par_names'])
    n_independent = len(cfg['model_cfg']['independent_par_names'])
    labels = (
        cfg['model_cfg']['shared_par_names'] +
        cfg['model_cfg']['independent_par_names'] +
        [cfg['model_cfg']['ratio_par_name']]
    )

    fx = sample_thetas[:, :n_shared]
    fig_pop_2 = None

    # Handling in case of independent parameters
    if n_independent > 0:
        fx = sample_thetas[:, :n_shared]
        fx = np.concatenate(
            (
                fx, sample_thetas[:, n_shared:-1:2]
            ), axis=-1
        )

        sx = sample_thetas[:, :n_shared]
        sx = np.concatenate(
            (
                sx,
                sample_thetas[:, n_shared+1:-1:2],
                sample_thetas[:,-1][:, None]
            ), axis=-1
        )

        fig_pop_2 = corner.corner(data=sx, color='r', **cfg['plot_cfg'])

    fx = np.concatenate(
        (
            fx, sample_thetas[:,-1][:, None]
        ), axis=-1
    )

    fig_pop_1 = corner.corner(data=fx, fig=fig_pop_2, labels=labels, **cfg['plot_cfg'])
    fig_pop_1.tight_layout()
    fig_pop_1.savefig(
        os.path.join(path, cfg['emcee_cfg']['run_name']+"_corner.pdf")
    )

    full_chain = backend.get_chain()
    par_names = (
        cfg['model_cfg']['shared_par_names'] +
        utils.gen_pop_par_names(
            cfg['model_cfg']['independent_par_names']
        ) +
        [cfg['model_cfg']['ratio_par_name']]
    )
    fig, axes = plt.subplots(
        ndim, figsize=(10, int(np.ceil(7*ndim/3))), sharex=True
    )
    for i in range(ndim):
        ax = axes[i]
        ax.plot(full_chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(full_chain))
        ax.set_ylabel(par_names[i])
    axes[-1].set_xlabel("step number");
    fig.tight_layout()
    if cfg['model_cfg']['use_sigmoid']:
        suffix = "_transformed"
    else:
        suffix = ""
    fig.savefig(
        os.path.join(path, cfg['emcee_cfg']['run_name']+suffix+"_walkers.pdf")
    )

if __name__ == "__main__":
    main()