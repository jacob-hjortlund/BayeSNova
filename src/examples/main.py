import hydra
import omegaconf
import os
import corner

import emcee as em
import numpy as np
import pandas as pd
import clearml as cl
import matplotlib.pyplot as plt
import scipy.stats as stats
import bayesnova.utils as utils
import bayesnova.model as model
import bayesnova.preprocessing as prep
import bayesnova.postprocessing as post

from time import time
from mpi4py import MPI
from itertools import combinations

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
    
    shared_host_galaxy_par_names = [None] * 2 * prep.n_shared_host_properties
    independent_host_galaxy_par_names = [None] * 2 * prep.n_independent_host_properties
    if cfg['model_cfg']['host_galaxy_cfg']['use_properties']:
        shared_host_galaxy_par_names[::2] = cfg['model_cfg']['host_galaxy_cfg']['shared_property_names']
        shared_host_galaxy_par_names[1::2] = [
            "sig_" + name for name in cfg['model_cfg']['host_galaxy_cfg']['shared_property_names']
        ]
        independent_host_galaxy_par_names[::2] = cfg['model_cfg']['host_galaxy_cfg']['independent_property_names']
        independent_host_galaxy_par_names[1::2] = [
            "sig_" + name for name in cfg['model_cfg']['host_galaxy_cfg']['independent_property_names']
        ]

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
            shared_host_galaxy_par_names, independent_host_galaxy_par_names,  
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
        tau = int(
            np.ceil(cfg['emcee_cfg']['n_steps'] / int(cfg['emcee_cfg']['tau_tol']))
        )
        print(f"\nUsing tau={tau} for tolerance {int(cfg['emcee_cfg']['tau_tol'])}\n")

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

    par_names = post.get_par_names(cfg)
    if using_MPI_and_is_master or using_multiprocessing or no_pool:

        par_quantiles, symmetrized_stds = post.map_mmap_comparison(
            sample_thetas, max_thetas, par_names, path, clearml_logger
        )

    print("\n-------- PARAM VALUES / ERRORS / Z-SCORES -----------\n")

    post.parameter_values(
        par_quantiles, symmetrized_stds, shared_host_galaxy_par_names,
        independent_host_galaxy_par_names, par_names, cfg, path, clearml_logger
    )

    print("\n----------------- PLOTS ---------------------\n")
    labels = (
        cfg['model_cfg']['shared_par_names'] +
        shared_host_galaxy_par_names +
        cfg['model_cfg']['independent_par_names'] +
        independent_host_galaxy_par_names +
        cfg['model_cfg']['cosmology_par_names'] +
        (not cfg['model_cfg']['use_physical_ratio']) * [cfg['model_cfg']['ratio_par_name']]
    )
    fig_path = os.path.join(path, "figures")
    os.makedirs(fig_path, exist_ok=True)

    membership_quantiles = post.get_membership_quantiles(
        backend, burnin, tau, cfg['model_cfg']
    )

    (
        cm, cm_norm, mapper, pop1_color, pop2_color
    ) = post.setup_colormap(membership_quantiles)

    post.corner_plot(
        sample_thetas, pop1_color, pop2_color, labels, cfg, fig_path
    )

    if cfg['model_cfg']['host_galaxy_cfg']['use_properties'] and prep.n_shared_host_properties > 0:
        n_shared = len(cfg['model_cfg']['shared_par_names'])
        shared_host_samples = sample_thetas[:, n_shared:n_shared+2*prep.n_shared_host_properties]
        labels = shared_host_galaxy_par_names
        n_bins = post.doane_bin_count(shared_host_samples)
        tmp_cfg = omegaconf.OmegaConf.to_container(cfg)
        tmp_cfg['plot_cfg']['corner_cfg']['hist_kwargs']['color'] = pop1_color
        fig_shared_host_pars = corner.corner(
            data=shared_host_samples, labels=labels,
            color=pop1_color, bins=n_bins,
            **tmp_cfg['plot_cfg']['corner_cfg']
        )
        fig_shared_host_pars.tight_layout()
        fig_shared_host_pars.savefig(
            os.path.join(fig_path, "shared_host_pars.png")
        )

    titles_list = ["All Observables",]
    if cfg['model_cfg']['host_galaxy_cfg']['use_properties'] and prep.n_independent_host_properties > 0:
        titles_list.append("Light Curve Observables")
        titles_list.append("All Host Properties")
        titles_list += utils.format_property_names(
            cfg['model_cfg']['host_galaxy_cfg']['independent_property_names']
        )
    
    post.membership_histogram(
        membership_quantiles, titles_list, mapper,
        prep.idx_calibrator_sn, cfg, fig_path
    )

    available_properties_names = data.keys()[::2]
    for property_name in available_properties_names:

        host_property = data[property_name].to_numpy()
        host_property_errors = data[property_name+"_err"].to_numpy()
        if np.all(host_property == prep.NULL_VALUE):
            continue
        
        host_property, host_property_errors = post.get_host_property_values(
            host_property, host_property_errors,
            prep.idx_unique_sn, prep.idx_duplicate_sn,
        )

        idx_not_calibrator, idx_calibrator = post.get_host_property_split_idx(
            host_property, host_property_errors,
            prep.idx_calibrator_sn
        )

        formatted_property_name = utils.format_property_names(
            [property_name], use_scientific_notation=False
        )[0]
        scientific_notation_property_name = utils.format_property_names(
            [property_name], use_scientific_notation=True
        )[0]
        print("\n----------------- HOST PROPERTY: ", formatted_property_name, "---------------------\n")

        for i in range(len(titles_list)):

            quantiles = membership_quantiles[i]
            idx_valid = quantiles[1,:] != prep.NULL_VALUE

            idx_not_calibrator_and_quantile = idx_not_calibrator & idx_valid
            idx_calibrator_and_quantile = idx_calibrator & idx_valid

            if (
                np.sum(idx_not_calibrator_and_quantile) + np.sum(idx_calibrator_and_quantile)
            ) == 0:
                continue

            membership_name = titles_list[i]
            is_log_mass = "log" in membership_name and "M_*" in membership_name
            is_log_ssfr = "log" in membership_name and "sSFR" in membership_name
            if is_log_mass:
                membership_name = "Mass"
            if is_log_ssfr:
                membership_name = "sSFR"

            print(f"\n{membership_name} membership\n")
            
            property_not_calibrator = host_property[idx_not_calibrator_and_quantile]
            property_calibrator = host_property[idx_calibrator_and_quantile]

            property_not_calibrator_errors = host_property_errors[idx_not_calibrator_and_quantile]
            property_calibrator_errors = host_property_errors[idx_calibrator_and_quantile]

            membership_not_calibrator = quantiles[1,idx_not_calibrator_and_quantile]
            membership_calibrator = quantiles[1,idx_calibrator_and_quantile]
            membership_not_calibrator_errors = np.abs(
                np.row_stack([
                    membership_not_calibrator - quantiles[0,idx_not_calibrator_and_quantile],
                    quantiles[2,idx_not_calibrator_and_quantile] - membership_not_calibrator
                ])
            )
            membership_calibrator_errors = np.abs(
                np.row_stack([
                    membership_calibrator - quantiles[0,idx_calibrator_and_quantile],
                    quantiles[2,idx_calibrator_and_quantile] - membership_calibrator
                ])
            )

            title = f"{membership_name} membership vs {formatted_property_name}"
            membership_y_axis = r"log$_{10}(p_1 / p_2)$"
            fig, ax = plt.subplots(figsize=(8, 8))
            fig, ax = post.scatter_plot(
                fig, ax,
                property_not_calibrator, membership_not_calibrator,
                property_not_calibrator_errors, membership_not_calibrator_errors.T,
                scientific_notation_property_name, membership_y_axis, membership_y_axis,
                membership_not_calibrator, cm, cm_norm, mapper, cfg, title
            )
            fig, ax = post.scatter_plot(
                fig, ax,
                property_calibrator, membership_calibrator,
                property_calibrator_errors, membership_calibrator_errors.T,
                scientific_notation_property_name, membership_y_axis, membership_y_axis,
                membership_calibrator, cm, cm_norm, mapper, cfg,
                title, edgecolor='k', use_colorbar=True, zorder=int(1e10)
            )

            save_path = os.path.join(
                fig_path,
                "host_property_vs_membership",
                f"{membership_name.replace(' ', '_')}"
            )
            os.makedirs(save_path, exist_ok=True)

            filename = formatted_property_name.replace(" ", "_")
            fig.savefig(os.path.join(save_path, filename+".png"))

            plt.close('all')


    membership_combinations = list(combinations(titles_list, 2))

    all_membership_quantiles = membership_quantiles[0]
    idx_not_calibrator = ~prep.idx_calibrator_sn
    idx_calibrator = prep.idx_calibrator_sn

    for membership_1, membership_2 in membership_combinations:
            
            quantiles_1 = membership_quantiles[titles_list.index(membership_1)]
            quantiles_2 = membership_quantiles[titles_list.index(membership_2)]

            idx_valid_1 = quantiles_1[1,:] != prep.NULL_VALUE
            idx_valid_2 = quantiles_2[1,:] != prep.NULL_VALUE
    
            idx_not_calibrator_and_quantile = idx_not_calibrator & idx_valid_1 & idx_valid_2
            idx_calibrator_and_quantile = idx_calibrator & idx_valid_1 & idx_valid_2

            n_not_calibrator = np.sum(idx_not_calibrator_and_quantile)
            n_calibrator = np.sum(idx_calibrator_and_quantile)
            n_total = n_not_calibrator + n_calibrator
            if n_total == 0:
                continue
            
            is_log_mass_1 = "log" in membership_1 and "M_*" in membership_1
            is_log_ssfr_1 = "log" in membership_1 and "sSFR" in membership_1
            is_local_1 = "Local" in membership_1
            is_global_1 = "Global" in membership_1
            if is_log_mass_1:
                membership_1 = is_local_1 * "Local " + is_global_1 * "Global " + "Mass"
            if is_log_ssfr_1:
                membership_1 = is_local_1 * "Local " + is_global_1 * "Global " "sSFR"

            is_log_mass_2 = "log" in membership_2 and "M_*" in membership_2
            is_log_ssfr_2 = "log" in membership_2 and "sSFR" in membership_2
            is_local_2 = "Local" in membership_2
            is_global_2 = "Global" in membership_2
            if is_log_mass_2:
                membership_2 = is_local_2 * "Local " + is_global_2 * "Global " "Mass"
            if is_log_ssfr_2:
                membership_2 = is_local_2 * "Local " + is_global_2 * "Global " "sSFR"

            print("\n----------------- MEMBERSHIP COMBINATION: ", membership_1, "vs", membership_2, "---------------------\n")
            print(f"\nNumber of calibrator SN: {n_calibrator}")
            print(f"Number of non-calibrator SN: {n_not_calibrator}")
            print(f"Total number of SN: {n_total}\n")
    
            membership_1_not_calibrator = quantiles_1[1,idx_not_calibrator_and_quantile]
            membership_1_calibrator = quantiles_1[1,idx_calibrator_and_quantile]
            membership_1_not_calibrator_errors = np.abs(
                np.row_stack([
                    membership_1_not_calibrator - quantiles_1[0,idx_not_calibrator_and_quantile],
                    quantiles_1[2,idx_not_calibrator_and_quantile] - membership_1_not_calibrator
                ])
            )
            membership_1_calibrator_errors = np.abs(
                np.row_stack([
                    membership_1_calibrator - quantiles_1[0,idx_calibrator_and_quantile],
                    quantiles_1[2,idx_calibrator_and_quantile] - membership_1_calibrator
                ])
            )
    
            membership_2_not_calibrator = quantiles_2[1,idx_not_calibrator_and_quantile]
            membership_2_calibrator = quantiles_2[1,idx_calibrator_and_quantile]
            membership_2_not_calibrator_errors = np.abs(
                np.row_stack([
                    membership_2_not_calibrator - quantiles_2[0,idx_not_calibrator_and_quantile],
                    quantiles_2[2,idx_not_calibrator_and_quantile] - membership_2_not_calibrator
                ])
            )
            membership_2_calibrator_errors = np.abs(
                np.row_stack([
                    membership_2_calibrator - quantiles_2[0,idx_calibrator_and_quantile],
                    quantiles_2[2,idx_calibrator_and_quantile] - membership_2_calibrator
                ])
            )

            all_membership_not_calibrator = all_membership_quantiles[1,idx_not_calibrator_and_quantile]
            all_membership_calibrator = all_membership_quantiles[1,idx_calibrator_and_quantile]

            title = f"{membership_1} membership vs {membership_2} membership"
            membership_x_axis = membership_1 + r" log$_{10}(p_1 / p_2)$"
            membership_y_axis = membership_2 + r" log$_{10}(p_1 / p_2)$"
            colorbar_label = r"All Observables log$_{10}(p_1 / p_2)$"
            fig, ax = plt.subplots(figsize=(8, 8))
            fig, ax = post.scatter_plot(
                fig, ax,
                membership_1_not_calibrator, membership_2_not_calibrator,
                membership_1_not_calibrator_errors.T, membership_2_not_calibrator_errors.T,
                membership_x_axis, membership_y_axis, colorbar_label,
                all_membership_not_calibrator, cm, cm_norm, mapper, cfg, title
            )
            fig, ax = post.scatter_plot(
                fig, ax,
                membership_1_calibrator, membership_2_calibrator,
                membership_1_calibrator_errors.T, membership_2_calibrator_errors.T,
                membership_x_axis, membership_y_axis, colorbar_label,
                all_membership_calibrator, cm, cm_norm, mapper, cfg, title,
                edgecolor='k', use_colorbar=True, zorder=int(1e10)
            )

            save_path = os.path.join(
                fig_path,
                "host_membership_vs_membership"
            )

            os.makedirs(save_path, exist_ok=True)

            filename = f"{membership_1.replace(' ', '_')}_vs_{membership_2.replace(' ', '_')}"
            fig.savefig(os.path.join(save_path, filename+".png"))

            plt.close('all')

    if cfg['model_cfg']['host_galaxy_cfg']['use_properties']:
        
        use_physical_ratio = cfg['model_cfg']['use_physical_ratio']
        independent_host_galaxy_property_names = cfg['model_cfg']['host_galaxy_cfg']['independent_property_names']
        shared_host_galaxy_property_names = cfg['model_cfg']['host_galaxy_cfg']['shared_property_names']
        n_independent_properties = len(independent_host_galaxy_property_names)
        n_host_galaxy_pars = 4 * n_independent_properties
        n_cosmology_pars = len(cfg['model_cfg']['cosmology_par_names'])
        
        idx_first = -(n_host_galaxy_pars + n_cosmology_pars + (not use_physical_ratio))
        idx_last = -n_cosmology_pars - (not use_physical_ratio)

        host_prop_par_samples = sample_thetas[:,idx_first:idx_last]

        w_1 = sample_thetas[:, -1]

        for i, host_property_name in enumerate(independent_host_galaxy_property_names):
            
            host_property_values = data[host_property_name].to_numpy()
            host_property_errors = data[host_property_name+"_err"].to_numpy()
            idx_observed = (
                (host_property_values == prep.NULL_VALUE) |
                (host_property_errors == prep.NULL_VALUE)
            )

            observed_host_property_values = host_property_values[~idx_observed]

            x_axis_max = np.max(observed_host_property_values)
            x_axis_min = np.min(observed_host_property_values)
            x_axis_max = (1 + np.sign(x_axis_max)*0.1)*x_axis_max
            x_axis_min = (1 - np.sign(x_axis_min)*0.1)*x_axis_min
            x_values = np.linspace(x_axis_min, x_axis_max, 1000)

            n_samples = cfg['plot_cfg'].get('n_samples', sample_thetas.shape[0])
            if n_samples == 'max':
                n_samples = sample_thetas.shape[0]
            n_samples = min(n_samples, sample_thetas.shape[0])

            host_property_name = utils.format_property_names(
                [host_property_name], use_scientific_notation=False
            )[0]
            print("\n-----------------", host_property_name.upper() ,"DISTRIBUTION ---------------------\n")

            pop_1_pdf_values = np.zeros((n_samples, len(x_values)))
            pop_2_pdf_values = np.zeros((n_samples, len(x_values)))

            idx_pop_1_mean = i*4
            idx_pop_2_mean = i*4 + 1
            idx_pop_1_std = i*4 + 2
            idx_pop_2_std = i*4 + 3

            for j in range(n_samples):
                pop_1_pdf_values[j] = stats.norm.pdf(
                    x_values, host_prop_par_samples[j,idx_pop_1_mean], host_prop_par_samples[j,idx_pop_1_std]
                )
                pop_2_pdf_values[j] = stats.norm.pdf(
                    x_values, host_prop_par_samples[j,idx_pop_2_mean], host_prop_par_samples[j,idx_pop_2_std]
                )

            combined_pdf_values = w_1[:,np.newaxis] * pop_1_pdf_values + (1 - w_1[:,np.newaxis]) * pop_2_pdf_values

            pop_1_quantiles = np.quantile(pop_1_pdf_values, [0.16, 0.5, 0.84], axis=0)
            pop_2_quantiles = np.quantile(pop_2_pdf_values, [0.16, 0.5, 0.84], axis=0)
            combined_quantiles = np.quantile(combined_pdf_values, [0.16, 0.5, 0.84], axis=0)

            fig, ax = plt.subplots(figsize=(8, 8))

            ax.hist(
                observed_host_property_values, density=True,
                color='k', alpha=0.5, label='Observed'
            )

            ax.plot(x_values, pop_1_quantiles[1,:], color=pop1_color, label='Pop 1')
            ax.fill_between(
                x_values, pop_1_quantiles[0,:], pop_1_quantiles[2,:],
                color=pop1_color, alpha=0.3
            )

            ax.plot(x_values, pop_2_quantiles[1,:], color=pop2_color, label='Pop 2')
            ax.fill_between(
                x_values, pop_2_quantiles[0,:], pop_2_quantiles[2,:],
                color=pop2_color, alpha=0.3
            )

            ax.plot(x_values, combined_quantiles[1,:], color='purple', label='Combined')
            ax.fill_between(
                x_values, combined_quantiles[0,:], combined_quantiles[2,:],
                color='purple', alpha=0.3
            )

            ax.set_xlabel(
                host_property_name, fontsize=cfg['plot_cfg']['label_kwargs']['fontsize']
            )
            ax.set_ylabel(
                "Probability Density", fontsize=cfg['plot_cfg']['label_kwargs']['fontsize']
            )
            ax.legend(fontsize=cfg['plot_cfg']['label_kwargs']['fontsize'])
            fig.tight_layout()

            save_path = os.path.join(
                fig_path,
                "host_property_distributions"
            )

            os.makedirs(save_path, exist_ok=True)

            filename = f"{host_property_name}.png"
            fig.savefig(os.path.join(save_path, filename))

            plt.close('all')

        shared_host_galaxy_property_names = cfg['model_cfg']['host_galaxy_cfg']['shared_property_names']
        n_shared_properties = len(shared_host_galaxy_property_names)
        n_host_galaxy_pars = 2 * n_shared_properties
        n_shared_pars = len(cfg['model_cfg']['shared_par_names'])
        
        idx_first = n_shared_pars
        idx_last = n_shared_pars + n_host_galaxy_pars

        host_prop_par_samples = sample_thetas[:,idx_first:idx_last]

        for i, host_property_name in enumerate(shared_host_galaxy_property_names):
            
            host_property_values = data[host_property_name].to_numpy()
            host_property_errors = data[host_property_name+"_err"].to_numpy()
            idx_observed = (
                (host_property_values == prep.NULL_VALUE) |
                (host_property_errors == prep.NULL_VALUE)
            )

            observed_host_property_values = host_property_values[~idx_observed]

            x_axis_max = np.max(observed_host_property_values)
            x_axis_min = np.min(observed_host_property_values)
            x_axis_max = (1 + np.sign(x_axis_max)*0.1)*x_axis_max
            x_axis_min = (1 - np.sign(x_axis_min)*0.1)*x_axis_min

            x_values = np.linspace(x_axis_min, x_axis_max, 1000)

            n_samples = cfg['plot_cfg'].get('n_samples', sample_thetas.shape[0])
            if n_samples == 'max':
                n_samples = sample_thetas.shape[0]
            n_samples = min(n_samples, sample_thetas.shape[0])

            host_property_name = utils.format_property_names(
                [host_property_name], use_scientific_notation=False
            )[0]
            print("\n-----------------", host_property_name.upper() ,"DISTRIBUTION ---------------------\n")

            pop_1_pdf_values = np.zeros((n_samples, len(x_values)))

            idx_pop_1_mean = i*2
            idx_pop_1_std = i*2 + 1

            for j in range(n_samples):
                pop_1_pdf_values[j] = stats.norm.pdf(
                    x_values, host_prop_par_samples[j,idx_pop_1_mean], host_prop_par_samples[j,idx_pop_1_std]
                )

            # mean_pop_1_pdf_values = np.mean(pop_1_pdf_values, axis=0)
            # std_pop_1_pdf_values = np.std(pop_1_pdf_values, axis=0)
            pop_1_quantiles = np.quantile(pop_1_pdf_values, [0.16, 0.5, 0.84], axis=0)

            fig, ax = plt.subplots(figsize=(8, 8))

            ax.hist(
                observed_host_property_values, density=True,
                color='k', alpha=0.5, label='Observed'
            )

            ax.plot(x_values, pop_1_quantiles[1,:], color=pop1_color)
            ax.fill_between(
                x_values, pop_1_quantiles[0,:], pop_1_quantiles[2,:],
                color=pop1_color, alpha=0.3
            )

            ax.set_xlabel(
                host_property_name, fontsize=cfg['plot_cfg']['label_kwargs']['fontsize']
            )
            ax.set_ylabel(
                "Probability Density", fontsize=cfg['plot_cfg']['label_kwargs']['fontsize']
            )
            ax.legend(fontsize=cfg['plot_cfg']['label_kwargs']['fontsize'])
            fig.tight_layout()

            save_path = os.path.join(
                fig_path,
                "host_property_distributions"
            )

            os.makedirs(save_path, exist_ok=True)

            filename = f"{host_property_name}.png"
            fig.savefig(os.path.join(save_path, filename))

            plt.close('all')

    print("Done!")

if __name__ == "__main__":
    main()