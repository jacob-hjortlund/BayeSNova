import hydra
import omegaconf
import sys
import os
import corner

import emcee as em
import numpy as np
import pandas as pd
import schwimmbad as swbd
import src.utils as utils
import src.preprocessing as prep
import src.probabilities as prob

from scipy.optimize import minimize
from time import time
from tqdm import tqdm

@hydra.main(
    version_base=None, config_path="configs", config_name="config"
)
def main(cfg: omegaconf.DictConfig) -> None:   

    # Setup results dir
    path = cfg['emcee_cfg']['save_path']
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
        sn_mb=sn_mb, sn_s=sn_s, sn_c=sn_c,
        sn_z=sn_z, lower_bound=cfg['model_cfg']['prior_lower_bound'],
        upper_bound=cfg['model_cfg']['prior_upper_bound'],
        n_workers=cfg['emcee_cfg']['n_workers']
    )

    t0 = time()
    with swbd.MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        # Print cfg
        print("\n----------------- CONFIG ---------------------\n")
        print(omegaconf.OmegaConf.to_yaml(cfg),"\n")

        print("\n----------------- SETUP ---------------------\n")
        # Setup init pos and log_prob
        theta = np.array([
                -0.14, 3.1, 3.7, 0.6, 2.9, -19.3, -19.3, -0.09, -0.09, 0.05, 0.03, -0.25, 0.1, 1.1, 1.1, 0.04, 0.04, 0.4
            ]) +3e-2 * np.random.rand(cfg['emcee_cfg']['n_walkers'],18)
        _ = log_prob(theta[0]) # call func once befor loop to jit compile
        nwalkers, ndim = theta.shape

        # Setup emcee backend
        filename = os.path.join(
            path, cfg['emcee_cfg']['file_name']
        )
        backend = em.backends.HDFBackend(filename, name=cfg['emcee_cfg']['run_name'])
        if not cfg['emcee_cfg']['continue_from_chain']:
            print("Resetting backend with name", cfg['emcee_cfg']['run_name'])
            backend.reset(nwalkers, ndim)

        print("\n----------------- SAMPLING ---------------------\n")
        sampler = em.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool, backend=backend)
        sampler.run_mcmc(theta, cfg['emcee_cfg']['n_steps'])
    
    t1 = time()
    total_time = t1-t0
    avg_eval_time = (total_time) / (cfg['emcee_cfg']['n_walkers'] * cfg['emcee_cfg']['n_steps'])
    print(f"\nTotal MCMC time:", total_time)
    print(f"Avg. time pr. step: {avg_eval_time} s\n")

    try:
        taus = sampler.get_autocorr_time(
            tol=cfg['emcee_cfg']['tau_tol']
        )
        tau = np.max(taus)
        print("Atuocorr lengths:\n", taus)
        print("Max autocorr length:", tau, "\n")
    except Exception as e:
        print("Got exception:\n")
        print(e, "\n")
        tau = cfg['emcee_cfg']['default_tau']
        print("\nUsing default tau:", tau, "\n")

    print("\nMax log(P):", np.max(sampler.get_log_prob(discard=5*tau, thin=int(0.5*tau), flat=True)), "\n")

    print("\n----------------- PLOTS ---------------------\n")

    samples = sampler.get_chain(discard=int(5*tau), thin=int(0.5*tau), flat=True)
    n_shared = len(cfg['model_cfg']['shared_par_names'])
    n_independent = len(cfg['model_cfg']['independent_par_names'])
    labels = (
        cfg['model_cfg']['shared_par_names'] +
        cfg['model_cfg']['independent_par_names'] +
        [cfg['model_cfg']['ratio_par_name']]
    )

    fx = samples[:, :n_shared]
    fig_pop_2 = None

    # Handling in case of independent parameters
    if n_independent > 0:
        fx = samples[:, :n_shared]
        fx = np.concatenate(
            (
                fx, samples[:, n_shared:-1:2]
            ), axis=-1
        )

        sx = samples[:, :n_shared]
        sx = np.concatenate(
            (
                sx,
                samples[:, n_shared+1:-1:2],
                samples[:,-1][:, None]
            ), axis=-1
        )

        fig_pop_2 = corner.corner(data=sx, color='r', **cfg['plot_cfg'])

    fx = np.concatenate(
        (
            fx, samples[:,-1][:, None]
        ), axis=-1
    )

    fig_pop_1 = corner.corner(data=fx, fig=fig_pop_2, labels=labels, **cfg['plot_cfg'])
    fig_pop_1.tight_layout()
    fig_pop_1.save_fig(
        os.path.join(path, cfg['emcee_cfg']['run_name'], "corner.pdf")
    )

if __name__ == "__main__":
    main()