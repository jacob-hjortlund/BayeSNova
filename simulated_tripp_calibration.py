import hydra
import omegaconf
import os
import tqdm
import time

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib.pyplot as plt
import emcee as em
import corner
import astropy.cosmology as cosmopy

import src.model as models
import src.preprocessing as prep
import src.utils as utils
from functools import partial

def dist_mod(observables, Mb, alpha, beta):
    mb = observables[:, 0]
    x1 = observables[:, 1]
    c = observables[:, 2]
    return mb - Mb + alpha * x1 - beta * c

def model_dist_mod(z, cosmo):
    return cosmo.distmod(z).value

def model_mag(observables, Mb, alpha, beta, cosmo):
    x1 = observables[:, 1]
    c = observables[:, 2]
    z = observables[:, 3]
    return cosmo.distmod(z).value + Mb - alpha * x1 + beta * c

def uncertainty(observables, covariance, pars, symm_errors):
    x1 = observables[:, 1]
    c = observables[:, 2]

    Mb = pars[0]
    alpha = pars[1]
    beta = pars[2]

    Mb_err = symm_errors[0]
    alpha_err = symm_errors[1]
    beta_err = symm_errors[2]
    
    var = (
        covariance[0,0] + Mb_err**2 + x1 * (covariance[1,1] + alpha_err**2) + 
        c * (covariance[2,2] + beta_err**2) - 2 * beta * covariance[0,2] +
        2 * alpha * covariance[0,1] - 2 * alpha * beta * covariance[1,2]
    )

    return np.sqrt(var)

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

@hydra.main(
    version_base=None, config_path="configs", config_name="tripp_config"
)
def main(cfg: omegaconf.DictConfig) -> None:

    global covariance
    global observables

    prompt_fractions = np.linspace(0.01, 0.99, 100)
    init_idx = cfg['emcee_cfg']['init_idx']
    n_runs = cfg['emcee_cfg']['n_runs']

    path = os.path.join(
        cfg['emcee_cfg']['save_path'],
        str(cfg['emcee_cfg']['sim_number']),
        cfg['emcee_cfg']['extra_path'],
        f"{init_idx}-{init_idx + n_runs}"
    )

    with utils.PoolWrapper(cfg['emcee_cfg']['pool_type']) as wrapped_pool:

        for i in range(n_runs):

            if wrapped_pool.is_mpi:
                wrapped_pool.check_if_master()

            t0 = time.time()

            prompt_fraction = prompt_fractions[init_idx+i]
            train_path = os.path.join(
                cfg['data_cfg']['train_path'],
                str(cfg['emcee_cfg']['sim_number']),
                f"prompt_fraction_{prompt_fraction:.2f}"
            )

            save_path = os.path.join(
                path, train_path.split(os.sep)[-1]
            )

            os.makedirs(save_path, exist_ok=True)

            # Init data
            data = pd.read_csv(
                os.path.join(train_path, "partial_sample.csv"),
                sep=cfg['data_cfg']['sep']
            )

            if cfg['emcee_cfg']['use_full_sample']:
                full = pd.read_csv(
                    os.path.join(train_path, "full_sample.csv"),
                    sep=cfg['data_cfg']['sep']
                )
                idx_pop_1 = full['true_class'].to_numpy() == 1.

                mb_full_1 = full['mb'].to_numpy()[idx_pop_1]
                x1_full_1 = full['x1'].to_numpy()[idx_pop_1]
                c_full_1 = full['c'].to_numpy()[idx_pop_1]
                z_full_1 = full['z'].to_numpy()[idx_pop_1]

                mb_full_2 = full['mb'].to_numpy()[~idx_pop_1]
                x1_full_2 = full['x1'].to_numpy()[~idx_pop_1]
                c_full_2 = full['c'].to_numpy()[~idx_pop_1]
                z_full_2 = full['z'].to_numpy()[~idx_pop_1]

            covariance = np.load(
                os.path.join(train_path, "covariance.npy"),
            )

            observables_keys = ['mb', 'x1', 'c', 'z']
            observables = data[observables_keys].to_numpy()

            mb = observables[:, 0]
            x1 = observables[:, 1]
            c = observables[:, 2]
            z = observables[:, 3]
            
            print(f"\n----------------------------Current prompt fraction: {prompt_fraction:.2f}\n-------------------------------\n")

            init_values = np.array([
                cfg['model_cfg']['init_values'][par] for par in cfg['model_cfg']['pars']
            ])

            # Init model and sample
            n_dim = len(cfg['model_cfg']['pars'])
            n_walkers = cfg['emcee_cfg']['n_walkers']
            tripp_model = models.TrippModel(cfg['model_cfg'])
            init_values = init_values[None, :] + 3e-2 * np.random.normal(size=(
                n_walkers, n_dim
            ))
            log_prob = partial(tripp_model, observables=observables, covariances=covariance)
            print("Constructing sampler...\n")
            sampler = em.EnsembleSampler(
                n_walkers, n_dim, log_prob, pool=wrapped_pool.pool
            )
            print("Running MCMC...\n")
            sampler.run_mcmc(init_values, cfg['emcee_cfg']['n_steps'])

            accept_frac = np.mean(sampler.acceptance_fraction)
            print("\nMean accepance fraction:", accept_frac, "\n")
            try:
                taus = sampler.get_autocorr_time(
                    tol=25
                )
                tau = np.max(taus)
                if np.isnan(tau):
                    raise Exception("Autocorr lengths are NaN")
                print("\nAtuocorr lengths:\n", taus)
                print("Max autocorr length:", tau, "\n")
            except Exception as e:
                print("\nGot exception:\n")
                print(e, "\n")
                tau = int(cfg['emcee_cfg']['n_steps'] / 25)
                print("\nUsing tolerance tau:", tau, "\n")

            t1 = time.time()
            print(f"\nMCMC took {t1-t0:.2f} seconds\n")
            t0 = time.time()

            burnin = int(5 * tau)
            thin = int(0.5 * tau)
            print(f"\nRemoving {burnin} steps as burnin")
            print(f"Thinning by {thin} steps\n")
            sample_thetas = sampler.get_chain(discard=burnin, thin=thin, flat=True)

            quantiles = np.quantile(
                sample_thetas, [0.16, 0.50, 0.84], axis=0
            )
            medians = quantiles[1, :]
            lower = medians - quantiles[0, :]
            upper = quantiles[2, :] - medians
            symm = (upper + lower) / 2

            tmp_df_array = np.concatenate(
                (medians, lower, upper, symm), axis=0
            )

            if i == 0:
                df_array = tmp_df_array
            else:
                df_array = np.row_stack((df_array, tmp_df_array))

            print("\nQuantiles:\n")
            for i, par in enumerate(cfg['model_cfg']['pars']):
                print(f"{par}: {medians[i]:.3f} +{upper[i]:.3f} -{lower[i]:.3f}")
            print("\n")

            fig = corner.corner(data=sample_thetas, labels=cfg['model_cfg']['pars'])
            fig.tight_layout()
            fig.savefig(
                os.path.join(save_path, "corner_plot.png")
            )

            z_plot = np.linspace(np.min(z), np.max(z), 1000)
            par_dict = tripp_model.input_to_dict(medians)
            cosmo=cosmopy.Flatw0waCDM(
                H0=par_dict['H0'], Om0=par_dict['Om0'],
                w0=par_dict['w0'], wa=par_dict['wa']
            )

            mu_tripp = dist_mod(
                observables, Mb=par_dict['Mb'],
                alpha=par_dict['alpha'], beta=par_dict['beta'],

            )
            mu_model = model_dist_mod(z, cosmo)
            mu_model_plot = model_dist_mod(z_plot, cosmo)
            residuals = mu_tripp - mu_model
            residual_errs = uncertainty(
                observables, covariance, medians, symm
            )

            mag_tripp = model_mag(
                observables, Mb=par_dict['Mb'],
                alpha=par_dict['alpha'], beta=par_dict['beta'],
                cosmo=cosmo
            )
            mag_residual = mb-mag_tripp
            mag_err = np.sqrt(covariance[0, 0])
            x1_err = np.ones(len(x1)) * np.sqrt(covariance[1, 1])
            c_err = np.ones(len(c)) * np.sqrt(covariance[2, 2])

            if cfg['emcee_cfg']['use_full_sample']:

                mb_tripp_full_1 = model_mag(
                    np.column_stack([mb_full_1, x1_full_1, c_full_1, z_full_1]),
                    Mb=par_dict['Mb'],
                    alpha=par_dict['alpha'], beta=par_dict['beta'],
                    cosmo=cosmo
                )

                mb_tripp_full_2 = model_mag(
                    np.column_stack([mb_full_2, x1_full_2, c_full_2, z_full_2]),
                    Mb=par_dict['Mb'],
                    alpha=par_dict['alpha'], beta=par_dict['beta'],
                    cosmo=cosmo
                )

                res_1 = mb_full_1 - mb_tripp_full_1
                res_2 = mb_full_2 - mb_tripp_full_2

                x1_min, x1_max = -3., 3.
                c_min, c_max = -0.3, 0.3
                res_min, res_max = -0.8, 0.8

                xx_x1, yy_resx = np.mgrid[x1_min:x1_max:100j, res_min:res_max:100j]
                x1_positions = np.vstack([xx_x1.ravel(), yy_resx.ravel()])
                kernel_x1_1 = stats.gaussian_kde(np.vstack([x1_full_1, res_1]))
                kernel_x1_2 = stats.gaussian_kde(np.vstack([x1_full_2, res_2]))
                f_x1_1 = np.reshape(kernel_x1_1(x1_positions).T, xx_x1.shape)
                f_x1_2 = np.reshape(kernel_x1_2(x1_positions).T, xx_x1.shape)

                xx_c, yy_resc = np.mgrid[c_min:c_max:100j, res_min:res_max:100j]
                c_positions = np.vstack([xx_c.ravel(), yy_resc.ravel()])
                kernel_c_1 = stats.gaussian_kde(np.vstack([c_full_1, res_1]))
                kernel_c_2 = stats.gaussian_kde(np.vstack([c_full_2, res_2]))
                f_c_1 = np.reshape(kernel_c_1(c_positions).T, xx_c.shape)
                f_c_2 = np.reshape(kernel_c_2(c_positions).T, xx_c.shape)

            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1]})
            ax[0].plot(z_plot, mu_model_plot, color='k')
            ax[0].errorbar(
                z, mu_tripp, yerr=residual_errs, fmt='none',
                color='r', capsize=0.5, capthick=0.5, elinewidth=0.5
            )
            ax[0].scatter(z, mu_tripp, color='r', s=1)
            ax[1].scatter(z, residuals, color='r', s=1)

            ax[0].set_ylabel(r"$\mu$", fontsize=24),
            ax[1].set_ylabel(r"$\mu_{\rm Tripp} - \mu_{\rm model}$", fontsize=16)
            ax[1].set_xlabel(r"$z$", fontsize=24)

            ax[0].set_xscale('log')

            fig.tight_layout()
            fig.savefig(
                os.path.join(save_path, "dist_mod.png")
            )

            fig, ax = plt.subplots(ncols=2, figsize=(8,4), sharey=True)
            ax[0].errorbar(
                x1, mag_residual, xerr=x1_err, yerr=mag_err, fmt='none',
                color='k', capsize=0.5, capthick=0.5, elinewidth=0.5, alpha=0.1
            )
            ax[0].scatter(x1, mag_residual, color='k', s=1, alpha=0.5)
            ax[0].set_xlabel(r"$x_1$", fontsize=16)
            ax[0].set_ylabel(r"$m_B - m_{B, \rm Tripp}$", fontsize=16)
            ax[1].errorbar(
                c, mag_residual, xerr=c_err, yerr=mag_err, fmt='none',
                color='k', capsize=0.5, capthick=0.5, elinewidth=0.5, alpha=0.1
            )
            ax[1].scatter(c, mag_residual, color='k', s=1, alpha=0.5)
            ax[1].set_xlabel(r"$c$", fontsize=16)

            if cfg['emcee_cfg']['use_full_sample']:
                cs_x1 = ax[0].contour(xx_x1, yy_resx, f_x1_1, colors='r')
                cs_c = ax[0].contour(xx_x1, yy_resx, f_x1_2, colors='b')
                cs_x1 = ax[1].contour(xx_c, yy_resc, f_c_1, colors='r')
                cs_c = ax[1].contour(xx_c, yy_resc, f_c_2, colors='b')

            fig.tight_layout()
            fig.savefig(
                os.path.join(save_path, "residuals.png")
            )

            t1 = time.time()
            print("\nPlots took: {:.2f} s\n".format(t1 - t0))

    df_array = np.concatenate(
        (df_array, prompt_fractions[init_idx:init_idx+n_runs, None]), axis=1
    )
    column_names = (
        cfg['model_cfg']['pars'] +
        [par + "_lower" for par in cfg['model_cfg']['pars']] +
        [par + "_upper" for par in cfg['model_cfg']['pars']] +
        [par + "_symm" for par in cfg['model_cfg']['pars']] +
        ['prompt_fraction']
    )
    df = pd.DataFrame(
        df_array, columns=column_names
    )

    df.to_csv(
        os.path.join(path, "results.csv")
    )

if __name__ == "__main__":
    main()