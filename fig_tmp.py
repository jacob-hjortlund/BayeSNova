import yaml
import dill
import pygtc
import numpy as np
import pandas as pd
import seaborn as sns
import jax_cosmo as jc
import jax.numpy as jnp
import matplotlib as mpl
import jax.random as random
import scipy.stats as stats
import matplotlib.pyplot as plt
import bayesnova.old_src.preprocessing as prep

from pathlib import Path
from flowjax.train import fit_to_data
from flowjax.distributions import Normal
from numpyro.infer import MCMC, NUTS, Predictive
from flowjax.bijections import RationalQuadraticSpline
from flowjax.flows import block_neural_autoregressive_flow, masked_autoregressive_flow
from bayesnova.numpyro_models import (
    sigmoid,
    SN,
    SNMass,
    SN2PopMass,
    tripp_mag,
    run_mcmc,
    SNTripp,
    SNTrippMass,
    distance_moduli,
)


def get_levels(Z, levels=[0.2, 0.4, 0.6, 0.8, 0.95, 0.99]):
    x = 1.0
    levels = levels.copy()
    Zmax = jnp.max(Z)
    Norm = jnp.sum(Z)
    lev = []
    for i in range(0, 1000):
        if np.sum(Z, where=Z > Zmax * x) / Norm > levels[0]:
            lev.append(x)
            levels.pop(0)
            if len(levels) == 0:
                break
        x = x * 0.99
    return jnp.asarray(lev) * Zmax


def map_to_latex(label: str):
    map_dict = {
        "H0": r"$H_0$",
        "f_host_1": r"$f_1^{\mathrm{host}}$",
        "f_sn_1": r"$f_1^{\mathrm{SN}}$",
        "f_SN_1_max": r"$f_{1,\mathrm{max}}^{\mathrm{SN}}$",
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


default_colors = sns.color_palette("colorblind")

COLOR = "white"
mpl.rcParams["text.color"] = COLOR
mpl.rcParams["axes.labelcolor"] = COLOR
mpl.rcParams["axes.edgecolor"] = COLOR
mpl.rcParams["xtick.color"] = COLOR
mpl.rcParams["ytick.color"] = COLOR

NUM_WARMUP = 500  # 10000
NUM_SAMPLES = 1000  # 75000
NUM_CHAINS = 1
F_SN_1_MIN = 0.15
VERBOSE = True
USE_RANGES = False
LOWER_HOST_MASS_BOUND = 6.0
UPPER_HOST_MASS_BOUND = 12.0
LOWER_R_B_bound = 1.5
UPPER_R_B_bound = 6.5
RNG_KEY, SUBKEY = random.split(random.PRNGKey(42))

DATA_NAME = "supercal_hubble_flow"
RUN_NAME = "Supercal_Hubble_Flow_Truncated_Positive_Scaling"
MODEL_NAME = "SN2PopMass"
MODEL = globals()[MODEL_NAME]
COSMOLOGY = jc.Planck15()

path = "/home/jacob/Uni/Msc/Thesis/BayeSNova"
# path = "/groups/dark/osman/BayeSNova"
base_path = Path(path)
data_path = base_path / "data"

output_path = base_path / "output" / MODEL_NAME / RUN_NAME
fig_path = output_path / "figures"
fig_path.mkdir(parents=True, exist_ok=True)
posterior_path = output_path / "posteriors"
posterior_path.mkdir(parents=True, exist_ok=True)

cfg_path = data_path / "config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["model_cfg"]["host_galaxy_cfg"]["use_properties"] = True
cfg["model_cfg"]["host_galaxy_cfg"]["independent_property_names"] = ["global_mass"]

# -------------------------- LOAD DATA --------------------------------- #

print("\nLoading data...\n")

data = pd.read_csv(data_path / (DATA_NAME + ".dat"), sep=" ")

prep.init_global_data(data, None, cfg["model_cfg"])

host_observables = prep.host_galaxy_observables.squeeze()
host_covariances = prep.host_galaxy_covariances.squeeze()

idx_valid_mass = host_observables != -9999.0
idx_not_calibrator = ~prep.idx_calibrator_sn
idx_valid = idx_not_calibrator

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    idx_valid = idx_valid & idx_valid_mass

sn_observables_np = prep.sn_observables[idx_valid]
sn_redshifts_np = prep.sn_redshifts[idx_valid]
sn_covariances_np = prep.sn_covariances[idx_valid]
host_observables_np = host_observables[idx_valid]
host_covariances_np = host_covariances[idx_valid]
host_uncertainty_np = np.sqrt(host_covariances_np)

v_disp = 250  # km/s
c = 3e5  # km/s
sn_covariances_np[:, 0, 0] += (5.0 / np.log(10) * v_disp / (c * sn_redshifts_np)) ** 2

sn_observables = jnp.array(sn_observables_np)
sn_redshifts = jnp.array(sn_redshifts_np)
sn_covariances = jnp.array(sn_covariances_np)
host_observables = jnp.array(host_observables_np)
host_uncertainty = jnp.array(host_uncertainty_np)
host_covariances = jnp.array(host_covariances_np)

host_mean = 0.0  # jnp.mean(host_observables, axis=0)
host_std = 1.0  # jnp.std(host_observables, axis=0)

with open(posterior_path / "combined_posterior_samples.pkl", "rb") as file:
    posterior_samples = dill.load(file)

# -------------------------- PLOT TRIANGLE --------------------------------- #

print("\nPlotting triangle...\n")

shared_params = [
    "alpha",
    "beta",
    "R_B",
    "R_B_scatter",
    "gamma_EBV",
    "scaling",
    "offset",
    "f_SN_1_max",
    "f_host_1",
]

if MODEL_NAME == "SN":
    shared_params += ["f_sn_1"]

independent_params = [
    "M_int",
    "X_int",
    "X_int_scatter",
    "C_int",
    "C_int_scatter",
    "tau_EBV",
]

if MODEL_NAME == "SNMass":
    shared_params += ["M_host", "M_host_scatter"]
elif MODEL_NAME == "SN2PopMass":
    independent_params += ["M_host", "M_host_scatter"]

cosmo_params = []

param_posterior_samples = []

pop_1_idx = []
pop_2_idx = []
param_labels = []

for param in cosmo_params + shared_params:
    if param not in posterior_samples.keys():
        continue
    param_posterior_samples.append(posterior_samples[param])
    param_labels.append(map_to_latex(param))

n_cosmo_and_shared = len(param_posterior_samples)
idx_counter = n_cosmo_and_shared
for param in independent_params:
    if param not in posterior_samples.keys():
        continue
    pop_1_idx.append(idx_counter)
    pop_2_idx.append(idx_counter + 1)
    param_posterior_samples.append(posterior_samples[param][:, 0])
    param_posterior_samples.append(posterior_samples[param][:, 1])
    param_labels.append(map_to_latex(param))
    idx_counter += 2

param_posterior_samples = np.array(param_posterior_samples).T

params_to_skip = [i for i in range(n_cosmo_and_shared)]
pop_1_idx = np.array(params_to_skip + pop_1_idx)
pop_2_idx = np.array(params_to_skip + pop_2_idx)

pop_1_samples = param_posterior_samples[:, pop_1_idx]
pop_2_samples = param_posterior_samples[:, pop_2_idx]

chain_labels = ["Population 2", "Population 1"]

print("Pop 1 Samples Shape: ", pop_1_samples.shape)
print("Pop 2 Samples Shape: ", pop_2_samples.shape)
print("No. of labels: ", len(param_labels))

param_ranges = [None] * len(param_labels)
if USE_RANGES:
    if MODEL_NAME == "SN":
        param_ranges[1] = (0, 6)
        param_ranges[2] = (2.4, 7.0)
        param_ranges[3] = (0, 2.5)
        param_ranges[4] = (1, 9)

    if MODEL_NAME == "SNMass":
        param_ranges[2] = (2, 9)
        param_ranges[3] = (0, 4)
        param_ranges[4] = (1, 3)

GTC = pygtc.plotGTC(
    chains=[pop_2_samples, pop_1_samples],
    paramNames=param_labels,
    chainLabels=chain_labels,
    nContourLevels=2,
    paramRanges=param_ranges,
    legendMarker="All",
    customLabelFont={"family": "Arial", "size": 15},
    customTickFont={"family": "Arial", "size": 6},
)

for axes in GTC.axes:
    for spine in axes.spines:
        axes.spines[spine].set_color("white")
    axes.tick_params(axis="both", colors="white")

GTC.savefig(fig_path / "gtc.png", transparent=True, dpi=300)

# -------------------------- FIT TRIPP MODEL TO OBSEVABLES --------------------------------- #

cosmo = jc.Planck15()
model_inputs = {
    "sn_observables": sn_observables,
    "sn_covariances": sn_covariances,
    "sn_redshifts": sn_redshifts,
    "host_mass": host_observables,
    "host_mass_err": host_uncertainty,
    "cosmology": cosmo,
    "verbose": False,
    "mass_cutoff": 10.0,
}

RNG_KEY, SUBKEY = random.split(SUBKEY)
SNTripp_mcmc = run_mcmc(
    rng_key=RNG_KEY,
    model=SNTripp,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    model_kwargs=model_inputs,
)

print("\nTripp Model Summary:\n")
SNTripp_mcmc.print_summary()
print("\n")

labels = [
    r"$M_{\mathrm{int}}$",
    r"$\sigma_{M_{\mathrm{int}}}$",
    r"$\alpha$",
    r"$\beta$",
]

SNTripp_samples = SNTripp_mcmc.get_samples()

tripp_posterior_samples = []
for key in SNTripp_samples.keys():
    tripp_posterior_samples.append(SNTripp_samples[key])
samples = np.array(tripp_posterior_samples).T

GTC = pygtc.plotGTC(
    chains=[samples],
    paramNames=labels,
    # nContourLevels=3,
)

for axes in GTC.axes:
    for spine in axes.spines:
        axes.spines[spine].set_color("white")
    axes.tick_params(axis="both", colors="white")

# GTC.tight_layout()
GTC.savefig(
    fig_path / "SNTripp_observed_data.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)


M_int_Tripp = jnp.percentile(SNTripp_samples["M_int"], q=50.0, axis=0)
alpha_Tripp = jnp.percentile(SNTripp_samples["alpha"], q=50.0, axis=0)
beta_Tripp = jnp.percentile(SNTripp_samples["beta"], q=50.0, axis=0)
print(f"M_int: {M_int_Tripp.shape}")
print(f"alpha: {alpha_Tripp.shape}")
print(f"beta: {beta_Tripp.shape}")

# -------------------------- SsMULATE SN OBSERVABLES --------------------------------- #

n_sim = 100000
sim_redshifts = random.uniform(RNG_KEY, minval=0.02, maxval=0.15, shape=(n_sim,))
print(f"Simulated redshifts: {sim_redshifts.shape}")

mean_sn_cov = jnp.mean(sn_covariances, axis=0)
sim_sn_covs = jnp.tile(mean_sn_cov, (n_sim, 1, 1))
print(f"Simulated SN Covariances: {sim_sn_covs.shape}")

mean_host_err = jnp.mean(host_uncertainty, axis=0)
sim_host_errs = jnp.tile(mean_host_err, (n_sim,))
print(f"Simulated Host Mass Errors: {sim_host_errs.shape}")

return_sites = [
    "X_int_latent_SN_pop_1",
    "X_int_latent_SN_pop_2",
    "X_int_latent",
    "C_int_latent_SN_pop_1",
    "C_int_latent_SN_pop_2",
    "C_int_latent",
    "EBV_latent_SN_pop_1",
    "EBV_latent_SN_pop_2",
    "EBV_latent",
    "EBV_latent_decentered",
    "R_B_latent",
    "R_B_latent_decentered",
    "sn_log_membership_ratio",
    "apparent_magnitude",
    "apparent_stretch",
    "apparent_color",
    "sn_observables",
]

if MODEL_NAME == "SNMass":
    return_sites += [
        "M_host_latent",
        "host_observables",
        "M_host_latent_decentered",
        "linear_function",
        "f_sn_1",
        "f_sn_2",
    ]

if MODEL_NAME == "SN2PopMass":
    return_sites += [
        "M_host_latent_pop_1",
        "M_host_latent_pop_2",
        "M_host_latent",
        "host_observables",
        "M_host_latent_decentered",
        "linear_function",
        "f_sn_1",
        "f_sn_2",
    ]

sim_posterior_samples = posterior_samples.copy()
for site in return_sites:
    if site in sim_posterior_samples.keys():
        sim_posterior_samples.pop(site)

for key in sim_posterior_samples.keys():
    sim_posterior_samples[key] = jnp.array(
        [jnp.percentile(sim_posterior_samples[key], q=50.0, axis=0)]
    )

predictive_inputs = {
    "sn_covariances": sim_sn_covs,
    "sn_redshifts": sim_redshifts,
    "host_mass_err": sim_host_errs,
    "host_mean": 0.0,
    "host_std": 1.0,
    "cosmology": COSMOLOGY,
    "lower_host_mass_bound": LOWER_HOST_MASS_BOUND,
    "upper_host_mass_bound": UPPER_HOST_MASS_BOUND,
    "lower_R_B_bound": LOWER_R_B_bound,
    "upper_R_B_bound": UPPER_R_B_bound,
}

RNG_KEY, SUBKEY = random.split(SUBKEY)
predictive = Predictive(
    MODEL, infer_discrete=False, posterior_samples=sim_posterior_samples
)  # , return_sites=return_sites)
simulated_supernovae = predictive(
    RNG_KEY,
    **predictive_inputs,
    # sn_covariances=sim_sn_covs,
    # sn_redshifts=sim_redshifts,
    # host_mass_err=sim_host_errs,
    # cosmology=COSMOLOGY,
    # host_mean=0.0,
    # host_std=1.0,
    # lower_host_mass_bound=LOWER_HOST_MASS_BOUND,
    # upper_host_mass_bound=UPPER_HOST_MASS_BOUND,
    # lower_R_B=LOWER_R_B_bound,
    # upper_R_B=UPPER_R_B_bound,
    # verbose=False,
)

simulated_sn_observables = simulated_supernovae["sn_observables"][0]
simulated_apparent_magnitudes = simulated_sn_observables[:, 0]
simulated_apparent_stretch = simulated_sn_observables[:, 1]
simulated_apparent_color = simulated_sn_observables[:, 2]

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    simulated_host_masses = (
        simulated_supernovae["host_observables"][0] * host_std + host_mean
    )

simulated_tripp_apparent_magnitude = tripp_mag(
    z=sim_redshifts,
    x1=simulated_apparent_stretch,
    c=simulated_apparent_color,
    M=M_int_Tripp,
    alpha=alpha_Tripp,
    beta=beta_Tripp,
    cosmology=COSMOLOGY,
)
simulated_hubble_residuals = (
    simulated_apparent_magnitudes - simulated_tripp_apparent_magnitude
)
simulated_absolute_magnitudes = simulated_apparent_magnitudes - distance_moduli(
    cosmology=COSMOLOGY, redshifts=sim_redshifts
)

observed_apparent_magnitude = sn_observables[:, 0]
observed_apparent_stretch = sn_observables[:, 1]
observed_apparent_color = sn_observables[:, 2]

observed_apparent_magnitude_errs = jnp.sqrt(sn_covariances[:, 0, 0])
observed_apparent_stretch_errs = jnp.sqrt(sn_covariances[:, 1, 1])
observed_apparent_color_errs = jnp.sqrt(sn_covariances[:, 2, 2])

observed_tripp_apparent_magnitude = tripp_mag(
    z=sn_redshifts,
    x1=observed_apparent_stretch,
    c=observed_apparent_color,
    M=M_int_Tripp,
    alpha=alpha_Tripp,
    beta=beta_Tripp,
    cosmology=COSMOLOGY,
)
observed_hubble_residuals = (
    observed_apparent_magnitude - observed_tripp_apparent_magnitude
)
observed_hubble_residual_errs = observed_apparent_magnitude_errs
observed_absolute_magnitudes = observed_apparent_magnitude - distance_moduli(
    cosmology=COSMOLOGY, redshifts=sn_redshifts
)

color_array = jnp.linspace(-0.3, 0.3, 1000)
stretch_array = jnp.linspace(-4, 4, 1000)
mass_array = jnp.linspace(6, 14, 1000)
residual_array = jnp.linspace(-0.5, 0.5, 1000)
magnitude_array = jnp.linspace(
    -0.25 + jnp.min(simulated_absolute_magnitudes),
    0.25 + jnp.max(simulated_absolute_magnitudes),
    1000,
)

color_residual_x_grid, color_residual_y_grid = jnp.meshgrid(color_array, residual_array)
stretch_residual_x_grid, stretch_residual_y_grid = jnp.meshgrid(
    stretch_array, residual_array
)
mass_residual_x_grid, mass_residual_y_grid = jnp.meshgrid(mass_array, residual_array)

color_positions = jnp.column_stack(
    [color_residual_x_grid.ravel(), color_residual_y_grid.ravel()]
)
stretch_positions = jnp.column_stack(
    [stretch_residual_x_grid.ravel(), stretch_residual_y_grid.ravel()]
)
mass_positions = jnp.column_stack(
    [mass_residual_x_grid.ravel(), mass_residual_y_grid.ravel()]
)

color_magnitude_x_grid, color_magnitude_y_grid = jnp.meshgrid(
    color_array, magnitude_array
)
stretch_magnitude_x_grid, stretch_magnitude_y_grid = jnp.meshgrid(
    stretch_array, magnitude_array
)
mass_magnitude_x_grid, mass_magnitude_y_grid = jnp.meshgrid(mass_array, magnitude_array)

color_magnitude_positions = jnp.column_stack(
    [color_magnitude_x_grid.ravel(), color_magnitude_y_grid.ravel()]
)
stretch_magnitude_positions = jnp.column_stack(
    [stretch_magnitude_x_grid.ravel(), stretch_magnitude_y_grid.ravel()]
)
mass_magnitude_positions = jnp.column_stack(
    [mass_magnitude_x_grid.ravel(), mass_magnitude_y_grid.ravel()]
)

# -------------------------- FIT MASS STEP TO OBSERVABLES --------------------------------- #

labels = [
    r"$M_{\mathrm{int}}$",
    r"$\sigma_{M_{\mathrm{int}}}$",
    r"$\alpha$",
    r"$\beta$",
    r"$\gamma$",
]

RNG_KEY, SUBKEY = random.split(SUBKEY)
SNTrippMass_observed_mcmc = run_mcmc(
    rng_key=RNG_KEY,
    model=SNTrippMass,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    model_kwargs=model_inputs,
)

print("\nObserved Tripp Mass Model Summary:\n")
SNTrippMass_observed_mcmc.print_summary()
print("\n")

SNTrippMass_observed_samples = SNTrippMass_observed_mcmc.get_samples()

SNTrippMass_observed_posterior_samples = []
for key in SNTrippMass_observed_samples.keys():
    SNTrippMass_observed_posterior_samples.append(SNTrippMass_observed_samples[key])
SNTrippMass_observed_samples = np.array(SNTrippMass_observed_posterior_samples).T

GTC = pygtc.plotGTC(
    chains=[SNTrippMass_observed_samples],
    paramNames=labels,
    # nContourLevels=3,
)

for axes in GTC.axes:
    for spine in axes.spines:
        axes.spines[spine].set_color("white")
    axes.tick_params(axis="both", colors="white")

# GTC.tight_layout()
GTC.savefig(
    fig_path / "SNTrippMass_observed_data.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)

# -------------------------- FIT MASS STEP TO SIMULATED DATA --------------------------------- #

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    N_SUBSAMPLE = 10000
    RNG_KEY, SUBKEY = random.split(SUBKEY)
    idx_subsample = random.choice(
        RNG_KEY,
        np.arange(simulated_sn_observables.shape[0]),
        shape=(N_SUBSAMPLE,),
        replace=False,
    )

    model_inputs = {
        "sn_observables": simulated_sn_observables[idx_subsample],
        "sn_covariances": sim_sn_covs[idx_subsample],
        "sn_redshifts": sim_redshifts[idx_subsample],
        "host_mass": simulated_host_masses[idx_subsample],
        "host_mass_err": sim_host_errs[idx_subsample],
        "cosmology": cosmo,
        "verbose": False,
        "mass_cutoff": 10.0,
    }

    RNG_KEY, SUBKEY = random.split(SUBKEY)
    SNTrippMass_simulated_mcmc = run_mcmc(
        rng_key=RNG_KEY,
        model=SNTrippMass,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        model_kwargs=model_inputs,
    )

    print("\nSimulated Tripp Mass Model Summary:\n")
    SNTrippMass_simulated_mcmc.print_summary()
    print("\n")

    SNTrippMass_simulated_samples = SNTrippMass_simulated_mcmc.get_samples()

    SNTrippMass_simulated_posterior_samples = []
    for key in SNTrippMass_simulated_samples.keys():
        SNTrippMass_simulated_posterior_samples.append(
            SNTrippMass_simulated_samples[key]
        )
    SNTrippMass_simulated_samples = np.array(SNTrippMass_simulated_posterior_samples).T

    GTC = pygtc.plotGTC(
        chains=[SNTrippMass_observed_samples, SNTrippMass_simulated_samples],
        paramNames=labels,
        chainLabels=["Observed", "Simulated"],
        legendMarker="All",
        nContourLevels=2,
    )

    for axes in GTC.axes:
        for spine in axes.spines:
            axes.spines[spine].set_color("white")
        axes.tick_params(axis="both", colors="white")

    # GTC.tight_layout()
    GTC.savefig(
        fig_path / "SNTrippMass_simulated_data.png",
        transparent=True,
        dpi=300,
        bbox_inches="tight",
    )

# -------------------------- PLOT POSTERIOR PREDICTIVE --------------------------------- #

print("\nPlotting posterior predictive...\n")
ncols = 3
if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    ncols = 4

fig, ax = plt.subplots(ncols=ncols, figsize=(12, 6), sharey=True)

ax[0].hist(
    observed_apparent_magnitude,
    density=True,
    bins="doane",
    color=default_colors[0],
    alpha=0.3,
    label="Observed",
)
ax[0].hist(
    simulated_apparent_magnitudes,
    density=True,
    bins="doane",
    color=default_colors[1],
    alpha=0.3,
    label="Simulated",
)
ax[0].set_xlabel(r"$m_{\mathrm{B,obs}}$", fontsize=25)
ax[0].set_ylabel(r"$p(m_{\mathrm{B,obs}})$", fontsize=25)
ax[0].legend(fontsize=15)

ax[1].hist(
    observed_apparent_stretch,
    density=True,
    bins="doane",
    color=default_colors[0],
    alpha=0.3,
)
ax[1].hist(
    simulated_apparent_stretch,
    density=True,
    bins="doane",
    color=default_colors[1],
    alpha=0.3,
)
ax[1].set_xlabel(r"$x_{1,\mathrm{app}}$", fontsize=25)

ax[2].hist(
    observed_apparent_color,
    density=True,
    bins="doane",
    color=default_colors[0],
    alpha=0.3,
)
ax[2].hist(
    simulated_apparent_color,
    density=True,
    bins="doane",
    color=default_colors[1],
    alpha=0.3,
)
ax[2].set_xlabel(r"$c_{\mathrm{app}}$", fontsize=25)

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    ax[3].hist(
        host_observables,
        density=True,
        bins="doane",
        color=default_colors[0],
        alpha=0.3,
    )
    ax[3].hist(
        simulated_host_masses,
        density=True,
        bins="doane",
        color=default_colors[1],
        alpha=0.3,
    )
    ax[3].set_xlabel(r"$M_{\mathrm{host}}$", fontsize=25)

fig.tight_layout()
fig.savefig(
    fig_path / "posterior_predictive.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)


# -------------------------- TRAIN RESIDUAL NORMALIZING FLOWS --------------------------------- #

FLOW_RNG_KEY, SUBKEY = random.split(SUBKEY)
TRAIN_KEY, SUBKEY = random.split(SUBKEY)
simulated_color_residuals = jnp.column_stack(
    [simulated_apparent_color, simulated_hubble_residuals]
)
base_dist = Normal(jnp.zeros(simulated_color_residuals.shape[1]))
flow = block_neural_autoregressive_flow(FLOW_RNG_KEY, base_dist=base_dist)
color_flow, losses = fit_to_data(
    key=TRAIN_KEY,
    dist=flow,
    x=simulated_color_residuals,
    learning_rate=1e-2,
)
color_pdf = jnp.reshape(
    jnp.exp(color_flow.log_prob(color_positions)), color_residual_x_grid.shape
)
color_levels = get_levels(color_pdf)


FLOW_RNG_KEY, SUBKEY = random.split(SUBKEY)
TRAIN_KEY, SUBKEY = random.split(SUBKEY)
simulated_stretch_residuals = jnp.column_stack(
    [simulated_apparent_stretch, simulated_hubble_residuals]
)
base_dist = Normal(jnp.zeros(simulated_stretch_residuals.shape[1]))
flow = block_neural_autoregressive_flow(FLOW_RNG_KEY, base_dist=base_dist)
stretch_flow, losses = fit_to_data(
    key=TRAIN_KEY,
    dist=flow,
    x=simulated_stretch_residuals,
    learning_rate=1e-2,
)
stretch_pdf = jnp.reshape(
    jnp.exp(stretch_flow.log_prob(stretch_positions)), stretch_residual_x_grid.shape
)
stretch_levels = get_levels(stretch_pdf)

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    FLOW_RNG_KEY, SUBKEY = random.split(SUBKEY)
    TRAIN_KEY, SUBKEY = random.split(SUBKEY)
    simulated_mass_residuals = jnp.column_stack(
        [simulated_host_masses, simulated_hubble_residuals]
    )
    base_dist = Normal(jnp.zeros(simulated_mass_residuals.shape[1]))
    flow = block_neural_autoregressive_flow(FLOW_RNG_KEY, base_dist=base_dist)
    mass_flow, losses = fit_to_data(
        key=TRAIN_KEY,
        dist=flow,
        x=simulated_mass_residuals,
        learning_rate=1e-2,
    )
    mass_pdf = jnp.reshape(
        jnp.exp(mass_flow.log_prob(mass_positions)), mass_residual_x_grid.shape
    )
    mass_levels = get_levels(mass_pdf)


# -------------------------- TRAIN MAGNITUDE NORMALIZING FLOWS --------------------------------- #

FLOW_RNG_KEY, SUBKEY = random.split(SUBKEY)
TRAIN_KEY, SUBKEY = random.split(SUBKEY)
simulated_color_magnitudes = jnp.column_stack(
    [simulated_apparent_color, simulated_absolute_magnitudes]
)
base_dist = Normal(jnp.zeros(simulated_color_magnitudes.shape[1]))
flow = block_neural_autoregressive_flow(FLOW_RNG_KEY, base_dist=base_dist)
color_magnitude_flow, losses = fit_to_data(
    key=TRAIN_KEY,
    dist=flow,
    x=simulated_color_magnitudes,
    learning_rate=1e-2,
)
color_magnitude_pdf = jnp.reshape(
    jnp.exp(color_magnitude_flow.log_prob(color_magnitude_positions)),
    color_magnitude_x_grid.shape,
)
color_magnitude_levels = get_levels(color_magnitude_pdf)


FLOW_RNG_KEY, SUBKEY = random.split(SUBKEY)
TRAIN_KEY, SUBKEY = random.split(SUBKEY)
simulated_stretch_magnitudes = jnp.column_stack(
    [simulated_apparent_stretch, simulated_absolute_magnitudes]
)
base_dist = Normal(jnp.zeros(simulated_stretch_magnitudes.shape[1]))
flow = block_neural_autoregressive_flow(FLOW_RNG_KEY, base_dist=base_dist)
stretch_magnitude_flow, losses = fit_to_data(
    key=TRAIN_KEY,
    dist=flow,
    x=simulated_stretch_magnitudes,
    learning_rate=1e-2,
)
stretch_magnitude_pdf = jnp.reshape(
    jnp.exp(stretch_magnitude_flow.log_prob(stretch_magnitude_positions)),
    stretch_magnitude_x_grid.shape,
)
stretch_magnitude_levels = get_levels(stretch_magnitude_pdf)

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    FLOW_RNG_KEY, SUBKEY = random.split(SUBKEY)
    TRAIN_KEY, SUBKEY = random.split(SUBKEY)
    simulated_mass_magnitudes = jnp.column_stack(
        [simulated_host_masses, simulated_absolute_magnitudes]
    )
    base_dist = Normal(jnp.zeros(simulated_mass_magnitudes.shape[1]))
    flow = block_neural_autoregressive_flow(FLOW_RNG_KEY, base_dist=base_dist)
    mass_magnitude_flow, losses = fit_to_data(
        key=TRAIN_KEY,
        dist=flow,
        x=simulated_mass_magnitudes,
        learning_rate=1e-2,
    )
    mass_magnitude_pdf = jnp.reshape(
        jnp.exp(mass_magnitude_flow.log_prob(mass_magnitude_positions)),
        mass_magnitude_x_grid.shape,
    )
    mass_magnitude_levels = get_levels(mass_magnitude_pdf)

# -------------------------- PLOT HUBBLE RESIDUALS --------------------------------- #

print("\nPlotting Hubble residuals...\n")

ncols = 2
if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    ncols = 3
    # mass_kde = stats.gaussian_kde(
    #     [simulated_host_masses, simulated_hubble_residuals],
    # )
    # mass_x_grid, mass_y_grid = np.mgrid[
    #     min(simulated_host_masses):max(simulated_host_masses):100j,
    #     min(simulated_hubble_residuals):max(simulated_hubble_residuals):100j
    # ]
    # mass_positions = np.vstack([mass_x_grid.ravel(), mass_y_grid.ravel()])
    # mass_kde_values = np.reshape(mass_kde(mass_positions).T, mass_x_grid.shape)

    # print("Mass KDE Complete...")

levels = 10  # [0.3935, 0.8647, 0.9889, 0.9997]#, 0.999996273346828]
fig, ax = plt.subplots(ncols=ncols, figsize=(12, 6), sharey=True)

# ax[0].contour(color_x_grid, color_y_grid, color_kde_values, levels=10)
ax[0].errorbar(
    observed_apparent_color,
    observed_hubble_residuals,
    xerr=observed_apparent_color_errs,
    yerr=observed_hubble_residual_errs,
    fmt="o",
    color=default_colors[0],
    alpha=0.3,
)
ax[0].contour(
    color_residual_x_grid,
    color_residual_y_grid,
    color_pdf,
    levels=np.flip(color_levels[:4]),
    linewidths=2,
    colors="white",
)
CS = ax[0].contour(
    color_residual_x_grid,
    color_residual_y_grid,
    color_pdf,
    levels=[color_levels[-1]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.99$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[0].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
CS = ax[0].contour(
    color_residual_x_grid,
    color_residual_y_grid,
    color_pdf,
    levels=[color_levels[-2]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.95$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[0].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
ax[0].set_xlabel(r"$c_{\mathrm{app}}$", fontsize=25)
ax[0].set_ylabel(r"$m_{\mathrm{B,obs}} - m_{\mathrm{B,Tripp}}$", fontsize=25)

# ax[1].contour(stretch_x_grid, stretch_y_grid, stretch_kde_values, levels=10)
ax[1].errorbar(
    observed_apparent_stretch,
    observed_hubble_residuals,
    xerr=observed_apparent_stretch_errs,
    yerr=observed_hubble_residual_errs,
    fmt="o",
    color=default_colors[0],
    alpha=0.3,
)
ax[1].contour(
    stretch_residual_x_grid,
    stretch_residual_y_grid,
    stretch_pdf,
    levels=np.flip(stretch_levels[:4]),
    linewidths=2,
    colors="white",
)
CS = ax[1].contour(
    stretch_residual_x_grid,
    stretch_residual_y_grid,
    stretch_pdf,
    levels=[stretch_levels[-1]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.99$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[1].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
CS = ax[1].contour(
    stretch_residual_x_grid,
    stretch_residual_y_grid,
    stretch_pdf,
    levels=[stretch_levels[-2]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.95$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[1].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
ax[1].set_xlabel(r"$x_{1,\mathrm{app}}$", fontsize=25)

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    # ax[2].contour(mass_x_grid, mass_y_grid, mass_kde_values, levels=10)
    ax[2].errorbar(
        host_observables_np,
        observed_hubble_residuals,
        xerr=host_uncertainty_np,
        yerr=observed_hubble_residual_errs,
        fmt="o",
        color=default_colors[0],
        alpha=0.3,
    )
    ax[2].contour(
        mass_residual_x_grid,
        mass_residual_y_grid,
        mass_pdf,
        levels=np.flip(mass_levels[:4]),
        linewidths=2,
        colors="white",
    )
    CS = ax[2].contour(
        mass_residual_x_grid,
        mass_residual_y_grid,
        mass_pdf,
        levels=[mass_levels[-1]],
        linewidths=2,
        colors="white",
    )
    strs = [r"$<0.99$"]
    fmt = {}
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    ax[2].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
    CS = ax[2].contour(
        mass_residual_x_grid,
        mass_residual_y_grid,
        mass_pdf,
        levels=[mass_levels[-2]],
        linewidths=2,
        colors="white",
    )
    strs = [r"$<0.95$"]
    fmt = {}
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    ax[2].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
    ax[2].set_xlabel(r"$M_{\mathrm{host}}$", fontsize=25)

fig.tight_layout()
fig.savefig(
    fig_path / "hubble_residuals.png", dpi=300, transparent=True, bbox_inches="tight"
)


# -------------------------- PLOT ABSOLUTE MAGNITUDE DIAGRAM --------------------------------- #

print("\nPlotting absolute magnitude diagram...\n")

ncols = 2
if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    ncols = 3

fig, ax = plt.subplots(ncols=ncols, figsize=(12, 6), sharey=True)

# ax[0].contour(color_x_grid, color_y_grid, color_kde_values, levels=10)
ax[0].errorbar(
    observed_apparent_color,
    observed_absolute_magnitudes,
    xerr=observed_apparent_color_errs,
    yerr=observed_apparent_magnitude_errs,
    fmt="o",
    color=default_colors[0],
    alpha=0.3,
)
ax[0].contour(
    color_magnitude_x_grid,
    color_magnitude_y_grid,
    color_magnitude_pdf,
    levels=np.flip(color_magnitude_levels[:4]),
    linewidths=2,
    colors="white",
)
CS = ax[0].contour(
    color_magnitude_x_grid,
    color_magnitude_y_grid,
    color_magnitude_pdf,
    levels=[color_magnitude_levels[-1]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.99$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[0].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
CS = ax[0].contour(
    color_magnitude_x_grid,
    color_magnitude_y_grid,
    color_magnitude_pdf,
    levels=[color_magnitude_levels[-2]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.95$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[0].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
ax[0].set_xlabel(r"$c_{\mathrm{app}}$", fontsize=25)
ax[0].set_ylabel(r"$M_{\mathrm{B}}$", fontsize=25)

# ax[1].contour(stretch_x_grid, stretch_y_grid, stretch_kde_values, levels=10)
ax[1].errorbar(
    observed_apparent_stretch,
    observed_absolute_magnitudes,
    xerr=observed_apparent_stretch_errs,
    yerr=observed_apparent_magnitude_errs,
    fmt="o",
    color=default_colors[0],
    alpha=0.3,
)
ax[1].contour(
    stretch_magnitude_x_grid,
    stretch_magnitude_y_grid,
    stretch_magnitude_pdf,
    levels=np.flip(stretch_magnitude_levels[:4]),
    linewidths=2,
    colors="white",
)
CS = ax[1].contour(
    stretch_magnitude_x_grid,
    stretch_magnitude_y_grid,
    stretch_magnitude_pdf,
    levels=[stretch_magnitude_levels[-1]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.99$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[1].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
CS = ax[1].contour(
    stretch_magnitude_x_grid,
    stretch_magnitude_y_grid,
    stretch_magnitude_pdf,
    levels=[stretch_magnitude_levels[-2]],
    linewidths=2,
    colors="white",
)
strs = [r"$<0.95$"]
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax[1].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
ax[1].set_xlabel(r"$x_{1,\mathrm{app}}$", fontsize=25)

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    # ax[2].contour(mass_x_grid, mass_y_grid, mass_kde_values, levels=10)
    ax[2].errorbar(
        host_observables_np,
        observed_absolute_magnitudes,
        xerr=host_uncertainty_np,
        yerr=observed_apparent_magnitude_errs,
        fmt="o",
        color=default_colors[0],
        alpha=0.3,
    )
    ax[2].contour(
        mass_magnitude_x_grid,
        mass_magnitude_y_grid,
        mass_magnitude_pdf,
        levels=np.flip(mass_magnitude_levels[:4]),
        linewidths=2,
        colors="white",
    )
    CS = ax[2].contour(
        mass_magnitude_x_grid,
        mass_magnitude_y_grid,
        mass_magnitude_pdf,
        levels=[mass_magnitude_levels[-1]],
        linewidths=2,
        colors="white",
    )
    strs = [r"$<0.99$"]
    fmt = {}
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    ax[2].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
    CS = ax[2].contour(
        mass_magnitude_x_grid,
        mass_magnitude_y_grid,
        mass_magnitude_pdf,
        levels=[mass_magnitude_levels[-2]],
        linewidths=2,
        colors="white",
    )
    strs = [r"$<0.95$"]
    fmt = {}
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    ax[2].clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=12)
    ax[2].set_xlabel(r"$M_{\mathrm{host}}$", fontsize=25)

fig.tight_layout()
fig.savefig(
    fig_path / "absolute_magnitude_diagram.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)

# -------------------------- PLOT POP FRACTION --------------------------------- #

print("\nPlotting population fraction...\n")

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass":
    f_SN_1_mid = posterior_samples["f_SN_1_mid"]
    scaling = posterior_samples["scaling"]
    # offset = posterior_samples["offset"]
    if MODEL_NAME == "SNMass":
        mean_mass = posterior_samples["M_host"]
        mass_scatter = posterior_samples["M_host_scatter"]
    elif MODEL_NAME == "SN2PopMass":
        mean_mass = posterior_samples["M_host_mixture_mean"]
        mass_scatter = posterior_samples["M_host_mixture_std"]

    mass = np.linspace(6, 13, 100)
    f_1_samples = np.zeros((NUM_SAMPLES, 100))
    for i in range(NUM_SAMPLES):
        obs_rescaled_mass = (mass - host_mean) / host_std
        rescaled_mass = (obs_rescaled_mass - mean_mass[i]) / mass_scatter[i]
        linear_function = scaling[i] * rescaled_mass  # + offset[i]
        f_1 = sigmoid(x=linear_function, f_mid=f_SN_1_mid[i], f_min=F_SN_1_MIN)
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
    fig.savefig(
        fig_path / ("sn_fraction.png"),
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
