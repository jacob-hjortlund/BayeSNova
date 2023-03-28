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
name="supercal+_simulation"
print("\nRunning", name, "\n")
data = pd.read_csv(
    #"./data/simulated_data/prompt_fraction_0.63/partial_sample.csv"
    "./data/simulated_data/supercal_constant_ratio_0.38/partial_sample.csv"
    #"./data/supercal_w_host_properties",
    #"./data/pantheon+_zcut",
    #sep=" "
)
full = pd.read_csv(
    "./data/simulated_data/supercal_constant_ratio_0.38/full_sample.csv"
)

covariance = np.load(
    "./data/simulated_data/supercal_constant_ratio_0.38/covariance.npy"
)

observables_keys = ['mb', 'x1', 'c', 'z']
sn_covariance_keys = ['x0', 'mBErr', 'x1Err', 'cErr', 'cov_x1_c', 'cov_x1_x0', 'cov_c_x0']
observables = data[observables_keys].to_numpy()
# sn_covariance_values = data[sn_covariance_keys].to_numpy()
# covariance, host_galaxy_covariances = prep.build_covariance_matrix(
#     sn_covariance_values, np.zeros((0,0))
# )

cfg = {
    "pars": [
        'Mb', 'alpha', 'beta', 'sig_int',
        #'Om0', 'w0'
    ],
    "preset_values": {
        'H0': 70.,
        'wa': 0.,
        'Om0': 0.3,
        'w0': -1.
    },
    "init_values": {
        'Mb': -19.3,
        'alpha': 0.128,
        'beta': 3.1,
        'sig_int': 0.12,
        'Om0': 0.3,
        'w0': -1.
    },
    "prior_bounds": {
        'Mb': {
            'lower': -23.,
            'upper': -15.
        },
        'alpha': {
            'lower':-10.,
            'upper': 10.
        },
        'beta': {
            'lower': -10.,
            'upper': 10.
        }, 
        'sig_int': {
            'lower': 0.00,
            'upper': 10.
        },
        'Om0': {
            'lower': 0.01,
            'upper': 0.99
        },
        'w0': {
            'lower': -3.,
            'upper': 1.
        }
    },
}

init_values = np.array([
    cfg['init_values'][par] for par in cfg['pars']
])

tripp_model = models.TrippModel(cfg)
np.random.seed(42)
n_walkers = 60
n_steps = 100
ndim = len(init_values)
init_values = init_values[None, :] + 3e-2 * np.random.normal(size=(
    n_walkers, ndim
))
sampler = em.EnsembleSampler(
    n_walkers, ndim, tripp_model, args=(observables, covariance)
)
sampler.run_mcmc(
    init_values, n_steps, progress=True
)

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
    tau = int(n_steps / 25)
    print("\nUsing default tau:", tau, "\n")

burnin = int(5 * tau)
print("Burn-in:", burnin, "\n")
sample_thetas = sampler.get_chain(discard=burnin, thin=int(0.5*tau), flat=True)

quantiles = np.quantile(
    sample_thetas, [0.16, 0.50, 0.84], axis=0
)
print("\nQuantiles:\n")
for i, par in enumerate(cfg['pars']):
    print(f"{par}: {quantiles[1, i]:.3f} +{quantiles[2, i] - quantiles[1, i]:.3f} -{quantiles[1, i] - quantiles[0, i]:.3f}")
print("\nEnd of", name, "\n")

fig = corner.corner(data=sample_thetas, labels=cfg['pars'])
fig.tight_layout()
fig.savefig(f"./{name}_tripp_calibration.png")

def dist_mod(observables, Mb, alpha, beta):
    mb = observables[:, 0]
    x1 = observables[:, 1]
    c = observables[:, 2]
    return mb - Mb + alpha * x1 - beta * c

def model_dist_mod(z, cosmo):
    return cosmo.distmod(z).value

def model_mag(observables, Mb, alpha, beta, cosmo):
    mb = observables[:, 0]
    x1 = observables[:, 1]
    c = observables[:, 2]
    z = observables[:, 3]
    return cosmo.distmod(z).value + Mb - alpha * x1 + beta * c

mb = observables[:, 0]
x1 = observables[:, 1]
c = observables[:, 2]
z = observables[:, 3]

idx_pop_1 = full['true_class'].to_numpy() == 1.

mb_full_1 = full['mb'].to_numpy()[idx_pop_1]
x1_full_1 = full['x1'].to_numpy()[idx_pop_1]
c_full_1 = full['c'].to_numpy()[idx_pop_1]
z_full_1 = full['z'].to_numpy()[idx_pop_1]

mb_full_2 = full['mb'].to_numpy()[~idx_pop_1]
x1_full_2 = full['x1'].to_numpy()[~idx_pop_1]
c_full_2 = full['c'].to_numpy()[~idx_pop_1]
z_full_2 = full['z'].to_numpy()[~idx_pop_1]

z_plot = np.linspace(np.min(z), np.max(z), 1000)
par_dict = tripp_model.input_to_dict(quantiles[1, :])
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

mag_tripp = model_mag(
    observables, Mb=par_dict['Mb'],
    alpha=par_dict['alpha'], beta=par_dict['beta'],
    cosmo=cosmo
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1]})
ax[0].plot(z_plot, mu_model_plot, color='k')
ax[0].scatter(z, mu_tripp, color='r', s=1)
ax[1].scatter(z, residuals, color='r', s=1)

ax[0].set_ylabel(r"$\mu$", fontsize=16),
ax[1].set_ylabel(r"$\mu_{\rm Tripp} - \mu_{\rm model}$", fontsize=16)
ax[1].set_xlabel(r"$z$", fontsize=16)

ax[0].set_xscale('log')

fig.tight_layout()
fig.savefig(f"./{name}_tripp_calibration_residuals.png")

plt.close('all')
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

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"
plt.close('all')

fig, ax = plt.subplots(ncols=2, figsize=(8,4))
cs_x1 = ax[0].contour(xx_x1, yy_resx, f_x1_1, colors='r')
cs_c = ax[0].contour(xx_x1, yy_resx, f_x1_2, colors='b')
ax[0].scatter(x1, mb-mag_tripp, color='k', alpha=0.2, s=1)
ax[0].set_xlabel(r"$x_1$", fontsize=16)
ax[0].set_ylabel(r"$m_B - m_{B, \rm Tripp}$", fontsize=16)
#ax[0].clabel(cs_x1, cs_x1.levels, inline=True, fmt=fmt, fontsize=10)
#ax[0].clabel(cs_c, cs_c.levels, inline=True, fmt=fmt, fontsize=10)

cs_x1 = ax[1].contour(xx_c, yy_resc, f_c_1, colors='r')
cs_c = ax[1].contour(xx_c, yy_resc, f_c_2, colors='b')
#ax[1].clabel(cs_x1, cs_x1.levels, inline=True, fmt=fmt, fontsize=10)
#ax[1].clabel(cs_c, cs_c.levels, inline=True, fmt=fmt, fontsize=10)
ax[1].scatter(c, mb-mag_tripp, color='k', alpha=0.2, s=1)
#ax[1].legend()
ax[1].set_xlabel(r"$c$", fontsize=16)
ax[1].set_ylabel(r"$m_B - m_{B, \rm Tripp}$", fontsize=16)
fig.tight_layout()
fig.savefig(f"./{name}_tripp_calibration_residuals_2d.png")