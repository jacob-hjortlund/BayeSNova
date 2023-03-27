import hydra
import omegaconf
import os
import corner

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import src.preprocessing as prep
import src.generative_models as gen

@hydra.main(
    version_base=None, config_path="configs", config_name="gen_config"
)
def main(cfg: omegaconf.DictConfig) -> None:

    base_data = pd.read_csv(
        cfg['data_cfg']['path'], sep=" "
    )

    volumetric_rates = None
    prep.init_global_data(base_data, volumetric_rates, cfg['model_cfg'])

    # Generate redshifts
    log10_z = np.log10(prep.sn_observables[:,-1])
    kde = stats.gaussian_kde(log10_z)
    kde_samples = kde.resample(
        cfg['simulation_cfg']['n_sne'], cfg['simulation_cfg']['seed']
    )
    z = 10**kde_samples[0]
    mean_observable_covariance = np.mean(prep.sn_covariances, axis=0)

    sn_observables_generator = gen.SNGenerator(cfg['model_cfg'])
    observables, true_classes, sn_rates = sn_observables_generator(z, mean_observable_covariance)

    fig, ax = plt.subplots(ncols=4, figsize=(20, 5))
    ax[0].hist(log10_z, bins=100, histtype='step', density=True, label='data')
    ax[0].hist(kde_samples[0], bins=100, histtype='step', density=True, label='sim')
    ax[0].set_xlabel('log10(z)')

    ax[1].hist(prep.sn_observables[:,0], bins=100, histtype='step', density=True, label='data')
    ax[1].hist(observables[:,0], bins=100, histtype='step', density=True, label='sim')
    ax[1].set_xlabel('mb')

    ax[2].hist(prep.sn_observables[:,1], bins=100, histtype='step', density=True, label='data')
    ax[2].hist(observables[:,1], bins=100, histtype='step', density=True, label='sim')
    ax[2].set_xlabel('x1')

    ax[3].hist(prep.sn_observables[:,2], bins=100, histtype='step', density=True, label='data')
    ax[3].hist(observables[:,2], bins=100, histtype='step', density=True, label='sim')
    ax[3].set_xlabel('c')

    fig.tight_layout()
    fig.savefig("./gen_no_z.png")

if __name__ == "__main__":
    main()