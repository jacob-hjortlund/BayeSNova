import jax
import yaml
import pickle
import arviz as az
import numpy as np
import pandas as pd
import numpyro as npy
import jax_cosmo as jc
import jax.numpy as jnp
import seaborn as sns
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import bayesnova.old_src.preprocessing as prep
from bayesnova.numpyro_models import SN, SNMass, SN2PopMass, run_mcmc

from jax import random
from pathlib import Path
from bayesnova.plot import corner
from numpyro.infer import MCMC, NUTS, Predictive

npy.util.enable_x64()


def sigmoid(x, scale):
    return scale / (1 + jnp.exp(-x))


def distance_moduli(cosmology, redshifts):
    scale_factors = jc.utils.z2a(redshifts)
    dA = jc.background.angular_diameter_distance(cosmology, scale_factors) / cosmology.h
    dL = (1.0 + redshifts) ** 2 * dA
    mu = 5.0 * jnp.log10(jnp.abs(dL)) + 25.0

    return mu


def tripp_mag(
    z,
    x1,
    c,
    cosmology,
    alpha=0.128,
    beta=3.00,
    M=-19.338,
):
    mu = distance_moduli(cosmology, z)

    return M + mu - alpha * x1 + beta * c


num_warmup = 10000
num_samples = 75000
num_chains = 1

path = Path("/home/jacob/Uni/Msc/Thesis/BayeSNova/data/")
fig_path = Path("/home/jacob/Uni/Msc/Thesis/BayeSNova/figures/")

sn_observables_np = np.load(path / "supercal_sn_observables.npy")
sn_redshifts_np = np.load(path / "supercal_sn_redshifts.npy")
sn_covariances_np = np.load(path / "supercal_sn_covariances.npy")
host_observables_np = np.load(path / "supercal_host_observables.npy")
host_uncertainty_np = np.load(path / "supercal_host_covariances.npy")

v_disp = 250  # km/s
c = 3e5  # km/s
sn_covariances_np[:, 0, 0] += (5.0 / np.log(10) * v_disp / (c * sn_redshifts_np)) ** 2

sn_observables = jnp.array(sn_observables_np)
sn_redshifts = jnp.array(sn_redshifts_np)
sn_covariances = jnp.array(sn_covariances_np)
host_observables = jnp.array(host_observables_np)
host_uncertainty = jnp.sqrt(jnp.array(host_uncertainty_np))

cosmology = jc.Planck15()

model_inputs = {
    "sn_observables": sn_observables,
    "sn_covariances": sn_covariances,
    "sn_redshifts": sn_redshifts,
    "host_mass": host_observables,
    "host_mass_err": host_uncertainty,
    "cosmology": cosmology,
    "verbose": False,
}

rng_key = random.PRNGKey(42)
rng_key, subkey = random.split(rng_key)

SNMass_mcmc = run_mcmc(
    model=SNMass,
    rng_key=rng_key,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    model_kwargs=model_inputs,
)


SNMass_posterior_samples = SNMass_mcmc.get_samples()
# Specify the file path where you want to save the pickled object
file_path = "/home/jacob/Uni/Msc/Thesis/BayeSNova/SNMass_posterior_samples.pkl"

# Pickle the posterior_samples object
with open(file_path, "wb") as file:
    pickle.dump(SNMass_posterior_samples, file)

SNMass_mcmc.print_summary()

f_1_max = SNMass_posterior_samples["f_1_max"]
scaling = SNMass_posterior_samples["scaling"]
offset = SNMass_posterior_samples["offset"]
mean_mass = SNMass_posterior_samples["M_host"]

mass = np.linspace(6, 13, 100)
f_1_samples = np.zeros((num_samples, 100))
for i in range(num_samples):
    linear_function = scaling[i] * (mass - mean_mass[i]) + offset[i]
    f_1 = sigmoid(linear_function, f_1_max[i])
    f_1_samples[i] = f_1

f_1_percentiles = np.percentile(f_1_samples, [16, 50, 84], axis=0)


default_colors = sns.color_palette("colorblind")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(mass, f_1_percentiles[1], color=default_colors[0], lw=3)
ax.fill_between(
    mass, f_1_percentiles[0], f_1_percentiles[2], color=default_colors[0], alpha=0.3
)
ax.set_xlabel(r"$\log_{10}(M_{\mathrm{host}})$", fontsize=25)
ax.set_ylabel(r"$f^{\mathrm{SN}}_1$", fontsize=25)
fig.tight_layout()
fig.savefig(fig_path / "sigmoid.png", dpi=300, transparent=True, bbox_inches="tight")


def map_to_latex(label: str):
    map_dict = {
        "H0": r"$H_0$",
        "f_sn_1": r"$f_1^{\mathrm{SN}}$",
        "f_1_max": r"$f_{1,\mathrm{max}}^{\mathrm{SN}}$",
        "M_int": r"$M_{\mathrm{int}}$",
        "M_int_scatter": r"$\sigma_{M_{\mathrm{int}}}$",
        "alpha": r"$\hat{\alpha}$",
        "X_int": r"$\hat{X}_{\mathrm{1,int}}$",
        "X_int_scatter": r"$\sigma_{X_{\mathrm{1,int}}}$",
        "beta": r"$\hat{\beta}$",
        "C_int": r"$\hat{c}_{\mathrm{int}}$",
        "C_int_scatter": r"$\sigma_{c_{\mathrm{int}}}$",
        "R_B": r"$\hat{R}_B$",
        "R_B_scatter": r"$\sigma_{R_B}$",
        "gamma_EBV": r"$\gamma_{\mathrm{E(B-V)}}$",
        "tau_EBV": r"$\tau_{\mathrm{E(B-V)}}$",
        "M_host": r"$\hat{M}_{\mathrm{host}}$",
        "M_host_scatter": r"$\sigma_{M_{\mathrm{host}}}$",
        "scaling": r"$a$",
        "offset": r"$b$",
    }

    return map_dict[label]


model = SNMass
mcmc = SNMass_mcmc
posterior_samples = SNMass_posterior_samples

rng_key, subkey = random.split(subkey)

predictive = Predictive(model, posterior_samples, infer_discrete=True)
discrete_samples = predictive(
    rng_key,
    sn_observables=sn_observables,
    sn_covariances=sn_covariances,
    sn_redshifts=sn_redshifts,
    host_mass=host_observables,
    host_mass_err=host_uncertainty,
    cosmology=cosmology,
)

chain_discrete_samples = jax.tree_util.tree_map(
    lambda x: x.reshape((1, num_samples) + x.shape[1:]), discrete_samples
)
mcmc.get_samples().update(discrete_samples)
mcmc.get_samples(group_by_chain=True).update(chain_discrete_samples)

pars_to_ignore = [
    "~population_assignment",
    "~sn_observables",
    "~host_observables",
    "~C_int_latent_SN_pop_1",
    "~C_int_latent_SN_pop_2",
    "~X_int_latent_SN_pop_1",
    "~X_int_latent_SN_pop_2",
    "~EBV_latent_SN_pop_1",
    "~EBV_latent_SN_pop_2",
    "~EBV_latent_decentered",
    "~R_B_latent_decentered",
    "~apparent_magnitude",
    "~apparent_color",
    " apparent_stretch",
    "~sn_log_membership_ratio",
    "~unshifted_gamma_EBV",
]

az.rcParams["plot.max_subplots"] = 200
az.plot_trace(
    az.from_numpyro(mcmc),
    compact=True,
    var_names=pars_to_ignore,
    backend_kwargs={"constrained_layout": True},
)
plt.tight_layout()
plt.savefig(fig_path / "SN_traceplot.png", dpi=300, bbox_inches="tight")

posterior_samples = SNMass_posterior_samples

shared_params = [
    "alpha",
    "beta",
    "R_B",
    "R_B_scatter",
    "gamma_EBV",
    "M_host",
    "M_host_scatter",
    "scaling",
    "offset",
    "f_1_max",
]
independent_params = [
    "M_int",
    "X_int",
    "X_int_scatter",
    "C_int",
    "C_int_scatter",
    "tau_EBV",
]

log10_pars = [
    "R_B_scatter",
    "M_host_scatter",
    "X_int_scatter",
    "C_int_scatter",
    "tau_EBV",
]
cosmo_params = []
n_cosmo = len(cosmo_params)
n_shared = len(shared_params)
n_independent = len(independent_params)

param_posterior_samples = []

pop_1_idx = []
pop_2_idx = []
param_labels = []

for param in cosmo_params + shared_params:
    if param not in posterior_samples.keys():
        continue
    if param in log10_pars:
        param_posterior_samples.append(np.log10(posterior_samples[param]))
    else:
        param_posterior_samples.append(posterior_samples[param])
    param_labels.append(map_to_latex(param))

idx_counter = n_cosmo + n_shared
for param in independent_params:
    if param not in posterior_samples.keys():
        continue
    pop_1_idx.append(idx_counter)
    pop_2_idx.append(idx_counter + 1)
    if param in log10_pars:
        param_posterior_samples.append(np.log10(posterior_samples[param][:, 0]))
        param_posterior_samples.append(np.log10(posterior_samples[param][:, 1]))
    else:
        param_posterior_samples.append(posterior_samples[param][:, 0])
        param_posterior_samples.append(posterior_samples[param][:, 1])
    param_labels.append(map_to_latex(param))
    idx_counter += 2

param_posterior_samples = np.array(param_posterior_samples).T

params_to_skip = [i for i in range(n_cosmo + n_shared)]
pop_1_idx = np.array(params_to_skip + pop_1_idx)
pop_2_idx = np.array(params_to_skip + pop_2_idx)

default_colors = sns.color_palette("colorblind")

fig = corner(
    param_posterior_samples[:, pop_2_idx],
    color=default_colors[0],
    labels=param_labels,
    label_kwargs={"fontsize": 30},
    hist2d_kwargs={"linewidth": 3.0, "nplot_2d": int(1e4), "q": 10.0},
    labelpad=0.25,
    max_n_ticks=3,
)
fig = corner(
    param_posterior_samples[:, pop_1_idx],
    color=default_colors[1],
    labels=param_labels,
    label_kwargs={"fontsize": 30},
    fig=fig,
    params_to_skip=params_to_skip,
    hist2d_kwargs={"linewidth": 3.0, "nplot_2d": int(1e4), "q": 10.0},
    labelpad=0.25,
    max_n_ticks=3,
)

fig.tight_layout()
fig.savefig(fig_path / "SNMass_corner.png", dpi=300, bbox_inches="tight")
