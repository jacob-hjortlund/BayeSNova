import hydra
import omegaconf

import emcee as em
import pandas as pd
import schwimmbad as swbd
import src.utils as utils
import src.preprocessing as prep
import src.probabilities as prob

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

    # Setup MCMC
    # How to initialize?

if __name__ == "__main__":
    main()