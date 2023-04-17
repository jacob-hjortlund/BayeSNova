import os
import time
import corner
import numpy as np
import pandas as pd
import emcee as em
import matplotlib.pyplot as plt

import src.utils as utils
import src.preprocessing as prep

from matplotlib.colors import Normalize

NULL_VALUE = -9999.

def mcmc_statistics(
    log_posterior, backend, tau: int, burnin: int,
    acceptance_fraction: float, save_path: str,
    clearml_logger
):
    sample_thetas = backend.get_chain(
        discard=burnin, thin=int(0.5 * tau), flat=True
    )
    sample_log_probs = backend.get_log_prob(
        discard=burnin, thin=int(0.5 * tau), flat=True
    )
    idx_max = np.argmax(sample_log_probs)
    max_sample_log_prob = sample_log_probs[idx_max]
    max_thetas = sample_thetas[idx_max]
    nlp = lambda *args: -log_posterior(*args)[0]
    # sol = sp_opt.minimize(nlp, max_thetas)
    
    #opt_pars = sol.x
    #opt_log_prob = -nlp(opt_pars)
    
    if not False:#sol.success:
        #print("Optimization not successful:", sol.message, "\n")
        print("Using init values corresponding to max sample log(P).")
        opt_log_prob = -nlp(max_thetas)
    
    print("Max sample log(P):", max_sample_log_prob)
    print("Optimized log(P):", opt_log_prob, "\n")

    # Log chain settings and log(P) values
    clearml_logger.report_single_value(
        name='acceptance_fraction', value=acceptance_fraction
    )
    clearml_logger.report_single_value(name='tau', value=tau)
    clearml_logger.report_single_value(name='burnin', value=burnin)
    clearml_logger.report_single_value(name="max_sample_log_prob", value=max_sample_log_prob)
    clearml_logger.report_single_value(name="opt_log_prob", value=opt_log_prob)

    # Save locally in dataframe
    metrics_df = pd.DataFrame(
        {
            "acceptance_fraction": [acceptance_fraction],
            "tau": [tau],
            "burnin": [burnin],
            "max_sample_log_prob": [max_sample_log_prob],
            "opt_log_prob": [opt_log_prob]
        }
    )
    metrics_df.to_csv(os.path.join(save_path, "metrics.csv"))

    return sample_thetas, max_thetas

def get_par_names(cfg: dict):

    if cfg['model_cfg']['host_galaxy_cfg']['use_properties']:
        host_galaxy_par_names = [
            host_gal_par_name for name in cfg['model_cfg']['host_galaxy_cfg']['property_names']
            for host_gal_par_name in ("mu_"+name, "sig_"+name)
        ]
    else:
        host_galaxy_par_names = []

    par_names = (
        cfg['model_cfg']['shared_par_names'] +
        utils.gen_pop_par_names(
            cfg['model_cfg']['independent_par_names']
        ) +
        utils.gen_pop_par_names(host_galaxy_par_names) +
        cfg['model_cfg']['cosmology_par_names'] +
        (not cfg['model_cfg']['use_physical_ratio']) * [cfg['model_cfg']['ratio_par_name']]
    )

    return par_names, host_galaxy_par_names

def map_mmap_comparison(
    sample_thetas: np.ndarray, max_thetas: np.ndarray,
    par_names: list, save_path: str, clearml_logger
):

    transposed_sample_thetas = sample_thetas.T

    quantiles = np.quantile(
        sample_thetas, [0.16, 0.50, 0.84], axis=0
    )
    symmetrized_stds = 0.5 * (quantiles[2] - quantiles[0])
    stds = np.std(sample_thetas, axis=0)
    map_mmap_values = np.zeros((5, len(par_names)))
    
    t2 = time.time()
    mmaps = np.array(list(map(utils.estimate_mmap, transposed_sample_thetas)))
    t3 = time.time()
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

    map_mmap_df.to_csv(os.path.join(save_path, "map_mmap.csv"))
    print(f"\nRMS value: {rms_value}\n")

    return quantiles, symmetrized_stds

def parameter_values(
    quantiles: np.ndarray, symmetrized_stds: np.ndarray,
    par_names: list, host_galaxy_par_names: list, 
    cfg: dict, save_path: str, clearml_logger
):

    val_errors = np.concatenate(
        (quantiles, symmetrized_stds[None, :]),
        axis=0
    )
    val_errors_df = pd.DataFrame(
        val_errors, index=['lower', 'median', 'upper', 'sym_sigma'], columns=par_names
    )

    idx_pop_params = quantiles.shape[1] - len(cfg['model_cfg']['cosmology_par_names']) - (not cfg['model_cfg']['use_physical_ratio'])
    max_symmetric_error = np.max(
        symmetrized_stds[
            len(cfg['model_cfg']['shared_par_names']):idx_pop_params
        ].reshape(-1, 2), axis=-1
    )
    param_diff = np.squeeze(
        np.abs(
            np.diff(
                quantiles[1][len(cfg['model_cfg']['shared_par_names']):idx_pop_params].reshape(-1, 2),
                axis=1
            )
        )
    )
    param_z_scores = param_diff / max_symmetric_error

    z_score_df = pd.DataFrame(
        param_z_scores[None,:], index=['Z'], columns=(
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

    val_errors_df.to_csv(os.path.join(save_path, "medians_and_errors.csv"))
    z_score_df.to_csv(os.path.join(save_path, "z_scores.csv"))

def get_membership_quantiles(
    backend: em.backends.HDFBackend, burnin: int, tau: int
):
    
    blobs = backend.get_blobs(
        discard=burnin, thin=int(0.5*tau), flat=True
    )
    log_full_membership_probs = blobs[:, 0, :]
    log_sn_membership_probs = blobs[:, 1, :]
    log_host_membership_probs = blobs[:, 2, :]
    full_membership_quantiles = np.quantile(
        log_full_membership_probs, [0.16, 0.50, 0.84], axis=0
    )
    sn_membership_quantiles = np.quantile(
        log_sn_membership_probs, [0.16, 0.50, 0.84], axis=0
    )
    host_membership_quantiles = np.quantile(
        log_host_membership_probs, [0.16, 0.50, 0.84], axis=0
    )

    return full_membership_quantiles, sn_membership_quantiles, host_membership_quantiles

def setup_colormap(
    full_membership_quantiles: np.ndarray, sn_membership_quantiles: np.ndarray,
    host_membership_quantiles: np.ndarray, cfg: dict, color_map: str = "coolwarm"
):

    cm_full_min, cm_full_max = (
        np.min(full_membership_quantiles[1,:]),
        np.max(full_membership_quantiles[1,:])
    )
    cm_sn_min, cm_sn_max = (
        np.min(sn_membership_quantiles[1,:]),
        np.max(sn_membership_quantiles[1,:])
    )
    cm_host_min, cm_host_max = (
        np.min(host_membership_quantiles[1,:]),
        np.max(host_membership_quantiles[1,:])
    )
    cm_min = np.min(
        [cm_full_min, cm_sn_min] +
        cfg['model_cfg']['host_galaxy_cfg']['use_properties'] * [cm_host_min]
    )
    cm_max = np.max(
        [cm_full_max, cm_sn_max] +
        cfg['model_cfg']['host_galaxy_cfg']['use_properties'] * [cm_host_max]
    )

    cm = plt.cm.get_cmap(color_map)
    cm_norm_full = Normalize(vmin=cm_min, vmax=cm_max, clip=True)
    mapper_full = plt.cm.ScalarMappable(norm=cm_norm_full, cmap=cm)
    pop2_color = mapper_full.to_rgba(cm_min)
    pop1_color = mapper_full.to_rgba(cm_max)

    return cm, cm_norm_full, mapper_full, pop1_color, pop2_color

def corner_plot(
    sample_thetas: np.ndarray, 
    pop1_color, pop2_color,
    labels: list,
    cfg: dict, save_path: str,
):
    
    n_shared = len(cfg['model_cfg']['shared_par_names'])
    n_independent = len(cfg['model_cfg']['independent_par_names'])
    idx_end_independent = sample_thetas.shape[1] - len(cfg['model_cfg']['cosmology_par_names']) - (not cfg['model_cfg']['use_physical_ratio'])
    extra_params_present = idx_end_independent != sample_thetas.shape[1]

    fx = sample_thetas[:, :n_shared]
    fig_pop_2 = None

    # Handling in case of independent parameters
    if n_independent > 0:
        fx = np.concatenate(
            (
                fx, sample_thetas[:, n_shared:idx_end_independent:2]
            ), axis=-1
        )

        sx_list = [
            sample_thetas[:, :n_shared],
            sample_thetas[:, n_shared+1:idx_end_independent:2]
        ]
        if extra_params_present:
            extra_params = sample_thetas[:, idx_end_independent:]
            n_extra_params = extra_params.shape[-1]
            extra_params = extra_params.reshape(-1, n_extra_params)
            sx_list += [extra_params]
        sx = np.concatenate(sx_list, axis=-1)

        fig_pop_2 = corner.corner(data=sx, color=pop2_color, **cfg['plot_cfg'])
    
    if extra_params_present:
        extra_params = sample_thetas[:, idx_end_independent:]
        n_extra_params = extra_params.shape[-1]
        extra_params = extra_params.reshape(-1, n_extra_params)
        fx_list = [fx, extra_params]
        fx = np.concatenate(fx_list, axis=-1)

    fig_pop_1 = corner.corner(data=fx, fig=fig_pop_2, color=pop1_color, labels=labels, **cfg['plot_cfg'])
    fig_pop_1.tight_layout()
    fig_pop_1.savefig(
        os.path.join(save_path, cfg['emcee_cfg']['run_name']+"_corner.png")
    )

def chain_plot(
    backend: em.backends.HDFBackend,
    burnin: int, par_names: list,
    cfg: dict, save_path: str,
):
    
    full_chain = backend.get_chain()
    ndim = full_chain.shape[-1]
    fig, axes = plt.subplots(
        ndim, figsize=(10, int(np.ceil(7*ndim/3))), sharex=True
    )
    if ndim == 1:
        axes = [axes]
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
    fig.suptitle("Walkers" + suffix, fontsize=int(2 * cfg['plot_cfg']['label_kwargs']['fontsize']))
    fig.savefig(
        os.path.join(save_path, cfg['emcee_cfg']['run_name']+suffix+"_walkers.png")
    )

def membership_histogram(
    quantiles_list: list, titles_list: list,
    mapper_full, idx_reordered_calibrator_sn: np.ndarray,
    cfg: dict, save_path: str,
):

    n_hists = len(quantiles_list)
    fig, ax = plt.subplots(
        ncols=n_hists, figsize=(15/3*n_hists, 5/3*n_hists)
    )
    for i, (membership_quantiles, title) in enumerate(
        zip(quantiles_list, titles_list)
    ):
        _, bins, patches = ax[i].hist(
            membership_quantiles[1,~idx_reordered_calibrator_sn],
            color="r",bins=20, density=True
        )
        bin_centers = 0.5*(bins[:-1]+bins[1:])

        for c, p in zip(bin_centers, patches):
            plt.setp(p, "facecolor", mapper_full.to_rgba(c))
        
        for j in range(
            np.count_nonzero(idx_reordered_calibrator_sn)
        ):
            value = membership_quantiles[1, idx_reordered_calibrator_sn][j]
            ax[i].axvline(
                value,
                color=mapper_full.to_rgba(value),
                linestyle="--",
                gapcolor="k"
            )

        ax[i].set_xlabel(
            r"log$_{10}\left(p_{pop 1}(SN)/p_{pop 2}(SN)\right)$",
            fontsize=cfg['plot_cfg']['label_kwargs']['fontsize']
        )
        ax[i].set_title(title, fontsize=cfg['plot_cfg']['label_kwargs']['fontsize'])
    fig.tight_layout()

    if cfg['model_cfg']['use_sigmoid']:
        suffix = "_transformed"
    else:
        suffix = ""
    fig.savefig(
        os.path.join(save_path, cfg['emcee_cfg']['run_name']+suffix+"_membership_hist.png")
    )

def observed_property_vs_membership(
    property_name: str, host_property: np.ndarray,
    host_property_errors: np.ndarray, membership_quantiles: np.ndarray,
    colormap, colormap_norm, mapper_full,
    idx_unique_sn: np.ndarray, idx_duplicate_sn: np.ndarray,
    idx_reordered_calibrator_sn: np.ndarray, cfg: dict, save_path: str,
):

    unique_host_properties, duplicate_host_properties = prep.reorder_duplicates(
        host_property, idx_unique_sn, idx_duplicate_sn
    )
    unique_host_properties_errors, duplicate_host_properties_errors = prep.reorder_duplicates(
        host_property_errors, idx_unique_sn, idx_duplicate_sn
    )

    duplicate_host_property_means = []
    duplicate_host_property_mean_errors = []
    for i, (duplicate_property, duplicate_property_error) in enumerate(
        zip(duplicate_host_properties, duplicate_host_properties_errors)
    ):
        idx_available = duplicate_property != NULL_VALUE
        if np.count_nonzero(idx_available) == 0:
            duplicate_host_property_means.append(NULL_VALUE)
            duplicate_host_property_mean_errors.append(NULL_VALUE)
            continue
        mean, err = utils.weighted_mean_and_error(
            duplicate_property[idx_available],
            duplicate_property_error[idx_available]
        )
        duplicate_host_property_means.append(mean)
        duplicate_host_property_mean_errors.append(err)
    
    duplicate_host_property_means = np.array(duplicate_host_property_means)
    duplicate_host_property_mean_errors = np.array(duplicate_host_property_mean_errors)
    
    host_property = np.concatenate(
        (unique_host_properties, duplicate_host_property_means)
    )
    host_property_errors = np.concatenate(
        (unique_host_properties_errors, duplicate_host_property_mean_errors)
    )

    hubble_flow_host, calibration_host = (
        host_property[~idx_reordered_calibrator_sn],
        host_property[idx_reordered_calibrator_sn]
    )
    hubble_flow_host_err, calibration_host_err = (
        host_property_errors[~idx_reordered_calibrator_sn],
        host_property_errors[idx_reordered_calibrator_sn]
    )
    idx_hubble_flow_observed = (
        (hubble_flow_host != NULL_VALUE) &
        (hubble_flow_host_err != NULL_VALUE)
    )
    idx_calibration_observed = (
        (calibration_host != NULL_VALUE) &
        (calibration_host_err != NULL_VALUE)
    )

    hubble_flow_host = hubble_flow_host[idx_hubble_flow_observed]
    calibration_host = calibration_host[idx_calibration_observed]
    hubble_flow_host_err = hubble_flow_host_err[idx_hubble_flow_observed]
    calibration_host_err = calibration_host_err[idx_calibration_observed]

    idx_hubble_flow_valid = np.ones_like(hubble_flow_host, dtype='bool') & (hubble_flow_host_err >= 0.) #np.abs(hubble_flow_host_err/hubble_flow_host) < 10.#0.25
    idx_calibration_valid = np.ones_like(calibration_host, dtype='bool') & (calibration_host_err >= 0.) #np.abs(calibration_host_err/calibration_host) < 10.#0.25
    hubble_flow_host, hubble_flow_host_err = hubble_flow_host[idx_hubble_flow_valid], hubble_flow_host_err[idx_hubble_flow_valid]
    calibration_host, calibration_host_err = calibration_host[idx_calibration_valid], calibration_host_err[idx_calibration_valid]
    print("No. of Hubble flow hosts: ", len(hubble_flow_host))
    print("No. of calibration hosts: ", len(calibration_host), "\n")

    hubble_flow_membership_quantiles = membership_quantiles[:, ~idx_reordered_calibrator_sn][:, idx_hubble_flow_observed][:, idx_hubble_flow_valid]
    calibration_membership_quantiles = membership_quantiles[:, idx_reordered_calibrator_sn][:, idx_calibration_observed][:, idx_calibration_valid]
    hubble_flow_medians = hubble_flow_membership_quantiles[1, :]
    calibration_medians = calibration_membership_quantiles[1, :]
    hubble_flow_errors = np.abs(
        np.row_stack([
            hubble_flow_medians - hubble_flow_membership_quantiles[0, :],
            hubble_flow_membership_quantiles[2, :] - hubble_flow_medians
        ])
    )
    calibration_errors = np.abs(
        np.row_stack([
            calibration_medians - calibration_membership_quantiles[0, :],
            calibration_membership_quantiles[2, :] - calibration_medians
        ])
    )
    
    hubble_flow_errorbar_colors = np.array([(mapper_full.to_rgba(p)) for p in hubble_flow_medians])
    calibration_errorbar_colors = np.array([(mapper_full.to_rgba(p)) for p in calibration_medians])

    fig, ax = plt.subplots()

    for x,y,ex,ey,color in zip(
        hubble_flow_host, hubble_flow_medians,
        hubble_flow_host_err, hubble_flow_errors.T, hubble_flow_errorbar_colors
    ):
        
        ey = ey.reshape(2,1)
        ax.errorbar(x, y, xerr=ex, yerr=ey, color=color, fmt='none', capsize=0.5, capthick=0.5, elinewidth=0.5)
    
    for x,y,ex,ey,color in zip(
        calibration_host, calibration_medians,
        calibration_host_err, calibration_errors.T, calibration_errorbar_colors
    ):
        
        ey = ey.reshape(2,1)
        ax.errorbar(x, y, xerr=ex, yerr=ey, color=color, fmt='none', capsize=0.5, capthick=0.5, elinewidth=0.5)
    
    ax.scatter(hubble_flow_host, hubble_flow_medians, c=hubble_flow_medians, cmap=colormap, norm=colormap_norm)
    if len(calibration_host) > 0:
        ax.scatter(
            calibration_host, calibration_medians, c=calibration_medians, cmap=colormap, norm=colormap_norm,
            edgecolors="k", zorder=1000
        )
    
    x_lower = cfg['plot_cfg']['property_ranges'][property_name]['lower']
    x_upper = cfg['plot_cfg']['property_ranges'][property_name]['upper']
    if property_name == "global_mass":
        ax.set_ylim(-5, 2.5)
    ax.set_xlim(x_lower, x_upper)
    ax.set_xlabel(property_name, fontsize=cfg['plot_cfg']['label_kwargs']['fontsize'])
    ax.set_ylabel(
        r"log$_{10}\left(p_{pop 1}(SN)/p_{pop 2}(SN)\right)$",
        fontsize=cfg['plot_cfg']['label_kwargs']['fontsize']
    )
    fig.tight_layout()

    if cfg['model_cfg']['use_sigmoid']:
        suffix = "_transformed"
    else:
        suffix = ""
    fig.savefig(
        os.path.join(save_path, cfg['emcee_cfg']['run_name']+suffix+"_"+property_name+"_vs_membership.png")
    )