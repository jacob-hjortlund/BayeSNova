import hydra
import omegaconf

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
        sn_z=sn_z
    )

    init_pos = np.array([
        -19.3,-0.14,3.1,-0.25, 1.1, -0.09, 0.05,3.7,0.04,0.6,2.9, -19.3,0.1, 1.1, -0.09,0.04,0.4, 0.03 
    ])
    theta = np.array([-0.14, 3.1, 3.7, 0.6, 2.9, -19.3, -19.3, -0.09, -0.09, 0.05, 0.03, -0.25, 0.1, 1.1, 1.1, 0.04, 0.04, 0.4])
    n_pars = len(theta)
    idx = np.empty(n_pars)

    for i in range(n_pars):
        tmp_idx = theta == init_pos[i]
        idx[tmp_idx] = i

    chains = np.load('./data/test.npz')['results'][100:499, idx.astype('int')]

    t_tot = 0.
    for chain_theta in tqdm(chains):
        t0 = time()
        _ = log_prob(chain_theta)
        t_tot += time()-t0
    print("Total time logged:", t_tot)
    print("Avg time pr. evaluation:", t_tot / len(chains))
    

    # # Setup MCMC
    # dims = len(cfg['model_cfg']['shared_par_names']) + 2 * len(cfg['model_cfg']['independent_par_names']) + 1
    # init = np.random.rand(100,dims)
    # sampler = em.EnsembleSampler(100, dims, log_prob)
    # sampler.run_mcmc(init, 1000, skip_initial_state_check=True, progress=True)

    print(1)
if __name__ == "__main__":
    main()