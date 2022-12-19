import hydra
import omegaconf
import sys

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

    # Print cfg
    print(cfg.pretty())    

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

    # call func once befor loop to jit compile
    t0 = time()
    with swbd.MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        theta = np.array([
            -0.14, 3.1, 3.7, 0.6, 2.9, -19.3, -19.3, -0.09, -0.09, 0.05, 0.03, -0.25, 0.1, 1.1, 1.1, 0.04, 0.04, 0.4
        ]) +3e-2 * np.random.rand(cfg['emcee_cfg']['n_walkers'],18)
        _ = log_prob(theta[0])
        nwalkers, ndim = theta.shape
        sampler = em.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        sampler.run_mcmc(theta, cfg['emcee_cfg']['n_steps'])
    t1 = time()
    total_time = t1-t0
    avg_eval_time = (total_time) / (cfg['emcee_cfg']['n_walkers'] * cfg['emcee_cfg']['n_steps'])
    print(f"Total time:", total_time)
    print(f"Avg. time pr. step: {avg_eval_time} s")
        
    samples = sampler.get_chain()
    np.savez_compressed('/groups/dark/osman/Thesis/data/baseline_full_chain',results=samples)
    flat_samples = sampler.get_chain(discard=500, thin=48, flat=True)
    print(flat_samples.shape)
    np.savez_compressed('/groups/dark/osman/Thesis/data/baseline_reduced_chain',results=flat_samples)
    tau = sampler.get_autocorr_time()
    print(tau)

if __name__ == "__main__":
    main()