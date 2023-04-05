import hydra
import omegaconf
import os
import tqdm

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import src.preprocessing as prep
import src.generative_models as gen
import src.cosmology_utils as cosmo_utils

@hydra.main(
    version_base=None, config_path="configs", config_name="gen_config"
)
def main(cfg: omegaconf.DictConfig) -> None:

    init_seed = cfg['simulation_cfg']['init_seed'] + cfg['simulation_cfg']['run_number']
    rng = np.random.default_rng(init_seed)
    for i in range(cfg['simulation_cfg']['n_seed_loops']):
        seed = rng.integers(1, 2**32 - 1, size=1)[0]
        rng = np.random.default_rng(seed)
    print(f"Seed: {seed}")
    base_data = pd.read_csv(
        cfg['data_cfg']['path'], sep=" "
    )

    volumetric_rates = None
    prep.init_global_data(base_data, volumetric_rates, cfg['model_cfg'])

    # Generate redshifts
    log10_z = np.log10(prep.sn_observables[:,-1])
    kde = stats.gaussian_kde(log10_z)
    kde_samples = kde.resample(
        cfg['simulation_cfg']['n_max'], seed
    )
    z = 10**kde_samples[0]
    idx_argsort = np.argsort(z)
    sorted_z = z[idx_argsort]
    mean_observable_covariance = np.mean(prep.sn_covariances, axis=0)


    cfg['model_cfg']['use_physical_ratio'] = cfg['simulation_cfg']['use_physical_ratio']
    if cfg['simulation_cfg']['use_physical_ratio']:
        idx_init = cfg['simulation_cfg']['idx_init']
        idx_final = idx_init + cfg['simulation_cfg']['number_of_fractions']
        prompt_fractions = np.linspace(
            cfg['simulation_cfg']['prompt_fraction_lower'],
            cfg['simulation_cfg']['prompt_fraction_upper'],
            cfg['simulation_cfg']['n_fraction_steps']
        )
        prompt_fractions = prompt_fractions[idx_init:idx_final]
    else:
        prompt_fractions = [cfg['simulation_cfg']['w']]
        cfg['model_cfg']['pars']['w'] = cfg['simulation_cfg']['w']

    # Generate observables
    model_cfg = omegaconf.OmegaConf.to_container(cfg['model_cfg'], resolve=True)
    model_cfg['seed'] = seed
    sn_observables_generator = gen.SNGenerator(model_cfg)

    for prompt_fraction in tqdm.tqdm(prompt_fractions):

        name = f'prompt_fraction_{prompt_fraction:.2f}'
        title = f'Prompt fraction: {prompt_fraction:.2f}'
        if not cfg['simulation_cfg']['use_physical_ratio']:
            name = f"constant_ratio_{cfg['simulation_cfg']['w']:.2f}"
            title = f"Constant ratio: {cfg['simulation_cfg']['w']:.2f}"
        path = os.path.join(
            cfg['simulation_cfg']['save_path'],
            str(cfg['simulation_cfg']['run_number']),
            name
        )
        os.makedirs(path, exist_ok=True)

        sn_observables_generator.cfg['pars']['cosmology']['prompt_fraction'] = prompt_fraction

        observables, true_classes, sn_rates = sn_observables_generator(z, mean_observable_covariance)
        if cfg['simulation_cfg']['use_physical_ratio']:
            prompt_probability = (sn_rates[:, 1] / sn_rates[:, 0])[idx_argsort]
            delayed_probability = (sn_rates[:, 2] / sn_rates[:, 0])[idx_argsort]
            sorted_sn_rates = sn_rates[idx_argsort]

        fig, ax = plt.subplots(ncols=4, figsize=(20, 5))
        fig.suptitle(title, fontsize=16)
        ax[0].hist(log10_z, bins=100, histtype='step', density=True, label='data')
        ax[0].hist(kde_samples[0], bins=100, histtype='step', density=True, label='sim')
        ax[0].set_xlabel(r'log$_{10}$(z)', fontsize=14)
        ax[0].legend(fontsize=14)

        ax[1].hist(prep.sn_observables[:,0], bins=100, histtype='step', density=True, label='data')
        ax[1].hist(observables[:,0], bins=100, histtype='step', density=True, label='sim')
        ax[1].set_xlabel('mb', fontsize=14)
        ax[1].legend(fontsize=14)

        ax[2].hist(prep.sn_observables[:,1], bins=100, histtype='step', density=True, label='data')
        ax[2].hist(observables[:,1], bins=100, histtype='step', density=True, label='sim')
        ax[2].set_xlabel('x1', fontsize=14)
        ax[2].legend(fontsize=14)

        ax[3].hist(prep.sn_observables[:,2], bins=100, histtype='step', density=True, label='data')
        ax[3].hist(observables[:,2], bins=100, histtype='step', density=True, label='sim')
        ax[3].set_xlabel('c', fontsize=14)
        ax[3].legend(fontsize=14)

        fig.tight_layout()
        fig.savefig(path + "/comparison_to_obs.png")

        if cfg['simulation_cfg']['use_physical_ratio']:
            fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
            fig.suptitle(title, fontsize=16)
            ax[0].plot(sorted_z, sorted_sn_rates[:, 0], label='Total')
            ax[0].plot(sorted_z, sorted_sn_rates[:, 1], label='Prompt')
            ax[0].plot(sorted_z, sorted_sn_rates[:, 2], label='Delayed')
            ax[0].set_xlabel('z', fontsize=14)
            ax[0].set_ylabel(r'N$_{Ia}$ [yr$^{-1}$ Mpc$^{-3}$]', fontsize=14)
            ax[0].set_yscale('log')
            ax[0].legend(fontsize=14)

            ax[1].plot(sorted_z, prompt_probability, label='Prompt')
            ax[1].plot(sorted_z, delayed_probability, label='Delayed')
            ax[1].set_xlabel('z', fontsize=14)
            ax[1].set_ylabel('Probability', fontsize=14)
            ax[1].legend(fontsize=14)

            fig.tight_layout()
            fig.savefig(path + "/rates.png")

        observables = np.concatenate(
            (z[:, None], observables, true_classes[:, None], sn_rates),
            axis=1
        )

        idx_zcut_lower = observables[:, 0] >= cfg['simulation_cfg']['z_cut_lower']
        idx_zcut_upper = observables[:, 0] <= cfg['simulation_cfg']['z_cut_upper']
        idx_scut = np.abs(observables[:, 2] < cfg['simulation_cfg']['s_cut'])
        idx_ccut = np.abs(observables[:, 3] < cfg['simulation_cfg']['c_cut'])
        idx_cut = idx_zcut_lower & idx_zcut_upper & idx_scut & idx_ccut

        partial_observables = observables[idx_cut]
        partial_observables = partial_observables[:cfg['simulation_cfg']['n_sne']]

        partial_df = pd.DataFrame(
            partial_observables,
            columns=[
                'z', 'mb', 'x1', 'c', 'true_class', 'total_rate',
                'prompt_rate', 'delayed_rate'
            ]
        )

        full_df = pd.DataFrame(
            observables,
            columns=[
                'z', 'mb', 'x1', 'c', 'true_class', 'total_rate',
                'prompt_rate', 'delayed_rate'
            ]
        )

        partial_df.to_csv(path + "/partial_sample.csv", index=False)
        full_df.to_csv(path + "/full_sample.csv", index=False)
        np.save(path + "/covariance.npy", mean_observable_covariance)

        plt.close('all')

if __name__ == "__main__":
    main()