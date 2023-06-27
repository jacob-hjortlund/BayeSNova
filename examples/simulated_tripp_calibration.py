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

import bayesnova.models_old as models_old
import bayesnova.preprocessing as prep
import bayesnova.utils as utils
from functools import partial
from scipy.ndimage import gaussian_filter

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
        covariance[:,0,0] + Mb_err**2 + x1 * (covariance[:,1,1] + alpha_err**2) + 
        c * (covariance[:,2,2] + beta_err**2) - 2 * beta * covariance[:,0,2] +
        2 * alpha * covariance[:,0,1] - 2 * alpha * beta * covariance[:,1,2]
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

    if cfg['model_cfg']['use_physical_ratios']:
        prompt_fractions = np.linspace(0.01, 0.99, 100)
    else:
        prompt_fractions = ['constant']
    init_idx = cfg['emcee_cfg']['init_idx']
    n_runs = cfg['emcee_cfg']['n_runs']

    path = os.path.join(
        cfg['emcee_cfg']['save_path'],
        str(cfg['emcee_cfg']['sim_number']),
        cfg['emcee_cfg']['extra_path'],
        f"{init_idx}-{init_idx + n_runs}"
    )

    column_names = (
        cfg['model_cfg']['pars'] +
        [par + "_lower" for par in cfg['model_cfg']['pars']] +
        [par + "_upper" for par in cfg['model_cfg']['pars']] +
        [par + "_symm" for par in cfg['model_cfg']['pars']] +
        ['prompt_fraction']
    )

    with utils.PoolWrapper(cfg['emcee_cfg']['pool_type']) as wrapped_pool:

        for i in range(n_runs):

            if wrapped_pool.is_mpi:
                wrapped_pool.check_if_master()

            t0 = time.time()

            prompt_fraction = prompt_fractions[init_idx+i]
            
            if cfg['data_cfg']['train_is_sim']:
                train_path = os.path.join(
                    cfg['data_cfg']['train_path'],
                    str(cfg['emcee_cfg']['sim_number']),
                    f"prompt_fraction_{prompt_fraction:.2f}"
                )

                train_data_path = os.path.join(
                    train_path, "partial_sample.csv"
                )
                save_path = os.path.join(
                    path, train_path.split(os.sep)[-1]
                )
            else:
                train_path = cfg['data_cfg']['train_path']
                train_data_path = train_path
                save_path = path


            os.makedirs(save_path, exist_ok=True)

            # Init data
            print(train_data_path)
            data = pd.read_csv(
                train_data_path,
                sep=cfg['data_cfg']['train_sep']
            )

            if cfg['emcee_cfg']['use_full_sample']:
                full_path = cfg['data_cfg'].get(
                    'full_path',                    
                    os.path.join(train_path, "full_sample.csv")
                )
                full = pd.read_csv(
                    full_path, sep=cfg['data_cfg']['full_sep']
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

            if cfg['data_cfg']['train_is_sim']:
                covariance = np.load(
                    os.path.join(train_path, "covariance.npy"),
                )

                observables_keys = ['mb', 'x1', 'c', 'z']
                observables = data[observables_keys].to_numpy()

                mb = observables[:, 0]
                x1 = observables[:, 1]
                c = observables[:, 2]
                z = observables[:, 3]
            else:
                prep.init_global_data(data, None, cfg['model_cfg'])
                mb = prep.sn_observables[:, 0]
                x1 = prep.sn_observables[:, 1]
                c = prep.sn_observables[:, 2]
                z = prep.sn_redshifts
                observables = np.concatenate(
                    [prep.sn_observables, prep.sn_redshifts[:, None]], axis=1
                )
                covariance = prep.sn_covariances
            
            print(f"\n----------------------------Current prompt fraction: {prompt_fraction}\n-------------------------------\n")

            init_values = np.array([
                cfg['model_cfg']['init_values'][par] for par in cfg['model_cfg']['pars']
            ])

            # Init model and sample
            n_dim = len(cfg['model_cfg']['pars'])
            n_walkers = cfg['emcee_cfg']['n_walkers']
            tripp_model = models_old.TrippModel(cfg['model_cfg'])
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
                tau = int(
                    np.ceil(cfg['emcee_cfg']['n_steps'] / 25)
                )
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
                (medians, lower, upper, symm, [prompt_fraction]), axis=0
            )

            does_results_dataframe_exist = os.path.exists(
                os.path.join(path, "results.csv")
            )
            if not does_results_dataframe_exist:
                df = pd.DataFrame(
                    tmp_df_array[None, :], columns=column_names
                )
                df.to_csv(
                    os.path.join(path, "results.csv")
                )
            else:
                df = pd.read_csv(
                    os.path.join(path, "results.csv")
                )
                df = pd.concat(
                    [df, pd.DataFrame(tmp_df_array[None,:], columns=column_names)]
                ).reset_index(drop=True)
                df.to_csv(
                    os.path.join(path, "results.csv")
                )

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

            if covariance.shape[0] == 2:
                covariance = np.tile(covariance, (observables.shape[0], 1, 1))

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
            mag_err = np.sqrt(covariance[:, 0, 0])
            x1_err = np.sqrt(covariance[:,1, 1])
            c_err = np.sqrt(covariance[:,2, 2])

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

                bins = [100, 100]
                
                res_1_range = [res_1.min(), res_1.max()]
                res_2_range = [res_2.min(), res_2.max()]
                x1_1_range = [x1_full_1.min(), x1_full_1.max()]
                x1_2_range = [x1_full_2.min(), x1_full_2.max()]
                c_1_range = [c_full_1.min(), c_full_1.max()]
                c_2_range = [c_full_2.min(), c_full_2.max()]

                bin_x1_1 = np.linspace(x1_1_range[0], x1_1_range[1], bins[0] + 1)
                bin_x1_2 = np.linspace(x1_2_range[0], x1_2_range[1], bins[0] + 1)
                bin_c_1 = np.linspace(c_1_range[0], c_1_range[1], bins[0] + 1)
                bin_c_2 = np.linspace(c_2_range[0], c_2_range[1], bins[0] + 1)
                bin_res_1 = np.linspace(res_1_range[0], res_1_range[1], bins[1] + 1)
                bin_res_2 = np.linspace(res_2_range[0], res_2_range[1], bins[1] + 1)

                levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 4.1, 0.5) ** 2)

                H_x1_1, x_x1_1, y_x1_1 = np.histogram2d(
                    x1_full_1, res_1, bins=[bin_x1_1, bin_res_1]
                )
                H_x1_2, x_x1_2, y_x1_2 = np.histogram2d(
                    x1_full_2, res_2, bins=[bin_x1_2, bin_res_2]
                )
                H_c_1, x_c_1, y_c_1 = np.histogram2d(
                    c_full_1, res_1, bins=[bin_c_1, bin_res_1]
                )
                H_c_2, x_c_2, y_c_2 = np.histogram2d(
                    c_full_2, res_2, bins=[bin_c_2, bin_res_2]
                )

                H_x1_1 = gaussian_filter(H_x1_1, 1.)
                H_x1_2 = gaussian_filter(H_x1_2, 1.)
                H_c_1 = gaussian_filter(H_c_1, 1.)
                H_c_2 = gaussian_filter(H_c_2, 1.)

                hists = [
                    [H_x1_1, x_x1_1, y_x1_1],
                    [H_x1_2, x_x1_2, y_x1_2],
                    [H_c_1, x_c_1, y_c_1],
                    [H_c_2, x_c_2, y_c_2]
                ]
                contours = []

                for hist in hists:
                    H, X, Y = hist
                    Hflat = H.flatten()
                    inds = np.argsort(Hflat)[::-1]
                    Hflat = Hflat[inds]
                    sm = np.cumsum(Hflat)
                    sm /= sm[-1]
                    V = np.empty(len(levels))
                    for i, v0 in enumerate(levels):
                        try:
                            V[i] = Hflat[sm <= v0][-1]
                        except IndexError:
                            V[i] = Hflat[0]
                    V.sort()
                    m = np.diff(V) == 0
                    while np.any(m):
                        V[np.where(m)[0][0]] *= 1.0 - 1e-4
                        m = np.diff(V) == 0
                    V.sort()

                    # Compute the bin centers.
                    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

                    # Extend the array for the sake of the contours at the plot edges.
                    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
                    H2[2:-2, 2:-2] = H
                    H2[2:-2, 1] = H[:, 0]
                    H2[2:-2, -2] = H[:, -1]
                    H2[1, 2:-2] = H[0]
                    H2[-2, 2:-2] = H[-1]
                    H2[1, 1] = H[0, 0]
                    H2[1, -2] = H[0, -1]
                    H2[-2, 1] = H[-1, 0]
                    H2[-2, -2] = H[-1, -1]
                    X2 = np.concatenate(
                        [
                            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
                            X1,
                            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
                        ]
                    )
                    Y2 = np.concatenate(
                        [
                            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
                            Y1,
                            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
                        ]
                    )
                    contours.append([X2.copy(), Y2.copy(), H2.copy()])

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
                ax[0].contour(contours[0][0], contours[0][1], contours[0][2].T, levels=levels, colors='r')
                ax[0].contour(contours[1][0], contours[1][1], contours[1][2].T, levels=levels, colors='b')
                ax[1].contour(contours[2][0], contours[0][1], contours[2][2].T, levels=levels, colors='r')
                ax[1].contour(contours[3][0], contours[1][1], contours[3][2].T, levels=levels, colors='b')

            fig.tight_layout()
            fig.savefig(
                os.path.join(save_path, "residuals.png")
            )

            t1 = time.time()
            print("\nPlots took: {:.2f} s\n".format(t1 - t0))

    # df_array = np.concatenate(
    #     (df_array, prompt_fractions[init_idx:init_idx+n_runs, None]), axis=1
    # )
    # column_names = (
    #     cfg['model_cfg']['pars'] +
    #     [par + "_lower" for par in cfg['model_cfg']['pars']] +
    #     [par + "_upper" for par in cfg['model_cfg']['pars']] +
    #     [par + "_symm" for par in cfg['model_cfg']['pars']] +
    #     ['prompt_fraction']
    # )
    # df = pd.DataFrame(
    #     df_array, columns=column_names
    # )

    # df.to_csv(
    #     os.path.join(path, "results.csv")
    # )

if __name__ == "__main__":
    main()