# -------------------------- IMPORTS --------------------------------- #

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import numpyro as npy

npy.util.enable_x64()
npy.util.set_platform("cpu")

import re
import jax
import dill
import yaml
import pygtc
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import jax_cosmo as jc
import jax.numpy as jnp
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt
import bayesnova.old_src.preprocessing as prep
import bayesnova.sampling as sampling

from jax import random
from pathlib import Path
from numpyro.handlers import do, seed, trace
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_sample
from numpyro.infer import MCMC, NUTS, Predictive
from bayesnova.plot import preprocess_samples, corner
from bayesnova.numpyro_models import (
    SN,
    SNDelta,
    SNMass,
    SNMassDelta,
    SNMassGP,
    SN2PopMass,
    SN2PopMassGP,
    run_mcmc,
    sigmoid,
    distance_moduli,
)

import blackjax
from jax.lax import cond
from fastprogress.fastprogress import progress_bar
from jax.experimental import host_callback

default_colors = sns.color_palette("colorblind")

COLOR = "white"
mpl.rcParams["text.color"] = COLOR
mpl.rcParams["axes.labelcolor"] = COLOR
mpl.rcParams["axes.edgecolor"] = COLOR
mpl.rcParams["xtick.color"] = COLOR
mpl.rcParams["ytick.color"] = COLOR

# -------------------------- HELPER FUNCTIONS --------------------------------- #


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


# -------------------------- SETTINGS --------------------------------- #

NUM_ADAPTATION = int(1e4)
MAX_SAMPLES = int(1e6)
NUM_SAMPLES = int(1e5)  # 10000000
TARGET_VARE = 5e-8  # 5e-4
SEED = 4928873
RNG_KEY = random.PRNGKey(SEED)

if NUM_SAMPLES > MAX_SAMPLES:
    SAMPLE_STEPS = np.full(NUM_SAMPLES // MAX_SAMPLES, MAX_SAMPLES)
    SAMPLE_STEPS[: NUM_SAMPLES % MAX_SAMPLES] += 1
else:
    SAMPLE_STEPS = np.array([NUM_SAMPLES])

NUM_CHAINS = 4
F_SN_1_MIN = 0.0
VERBOSE = False
CONTINUE = False
LOWER_HOST_MASS_BOUND = 6.0
UPPER_HOST_MASS_BOUND = 12.0
LOWER_R_B_bound = 1.5
UPPER_R_B_bound = 6.5

DATASET_NAME = "supercal_hubble_flow"
RUN_NAME = "test"  # "Supercal_MCLMC"
MODEL_NAME = "SNDelta"
MODEL = globals()[MODEL_NAME]
# MODEL = do(MODEL, {"f_SN_1": 0.35})


print("\nModel: ", MODEL_NAME)
print("Run Name: ", RUN_NAME)
print("Num Warmup: ", NUM_ADAPTATION)
print("Num Samples: ", NUM_SAMPLES)
print("Num Chains: ", NUM_CHAINS, "\n")

# path = "/home/jacob/Uni/Msc/Thesis/BayeSNova"
path = "/groups/dark/osman/BayeSNova/"
base_path = Path(path)
data_path = base_path / "data"

output_path = base_path / "output" / MODEL_NAME / RUN_NAME
output_path.mkdir(parents=True, exist_ok=True)
fig_path = output_path / "figures"
fig_path.mkdir(parents=True, exist_ok=True)
posterior_path = output_path / "posteriors"
posterior_path.mkdir(parents=True, exist_ok=True)

previous_runs = [
    previous_run.name for previous_run in posterior_path.glob("*[0-9].pkl")
]
previous_run_numbers = [
    int(digit)
    for previous_run in previous_runs
    for digit in re.findall(r"\d+(?=.pkl)", previous_run)
]

if CONTINUE and len(previous_run_numbers) > 0:
    last_run_number = max(previous_run_numbers)
    print(f"Continuing from run {last_run_number}...\n")
    PREVIOUS_RUN = "_" + str(last_run_number)
    RUN_MODIFIER = "_" + str(last_run_number + 1)
else:
    if len(previous_runs) == 0:
        print(
            "Set to continue but no previous runs found, check your settings. Starting from scratch...\n"
        )
    else:
        print("Starting from scratch...\n")
    CONTINUE = False
    RUN_MODIFIER = "_0"

cfg_path = data_path / "config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["model_cfg"]["host_galaxy_cfg"]["use_properties"] = True
cfg["model_cfg"]["host_galaxy_cfg"]["independent_property_names"] = ["global_mass"]

shared_params = [
    "alpha",
    "beta",
    "R_B",
    "R_B_scatter",
    "gamma_EBV",
    "scaling",
    "M_host",
    "M_host_scatter",
    "f_SN_1_max",
    "f_host_1",
    "M_int",
    "delta_M_int",
    "X_int",
    "delta_X_int",
    "X_int_scatter",
    "delta_X_int_scatter",
    "C_int",
    "delta_C_int",
    "C_int_scatter",
    "delta_C_int_scatter",
    "tau_EBV",
    "delta_tau_EBV",
]

if MODEL_NAME == "SN" or MODEL_NAME == "SNDelta":
    shared_params += ["f_sn_1"]

independent_params = []

# -------------------------- LOAD DATA --------------------------------- #

print("\nLoading data...\n")

data = pd.read_csv(data_path / (DATASET_NAME + ".dat"), sep=" ")

prep.init_global_data(data, None, cfg["model_cfg"])

host_observables = prep.host_galaxy_observables.squeeze()
host_covariances = prep.host_galaxy_covariances.squeeze()

idx_valid_mass = host_observables != -9999.0
idx_not_calibrator = ~prep.idx_calibrator_sn
idx_valid = idx_not_calibrator

if (
    MODEL_NAME == "SNMass"
    or MODEL_NAME == "SNMassDelta"
    or MODEL_NAME == "SN2PopMass"
    or MODEL_NAME == "SNMassGP"
    or MODEL_NAME == "SN2PopMassGP"
):
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

host_mean = 0.0  # jnp.mean(host_observables, axis=0)
host_std = 1.0  # jnp.std(host_observables, axis=0)

# ---------------------- PRIOR PREDECTIVE ----------------------------- #

print("\nSampling Prior Predictive...\n")

cosmology = jc.Planck15()

prior_inputs = {
    "sn_covariances": sn_covariances,
    "sn_redshifts": sn_redshifts,
    "host_mass_err": host_uncertainty,
    "cosmology": cosmology,
    "lower_host_mass_bound": LOWER_HOST_MASS_BOUND,
    "upper_host_mass_bound": UPPER_HOST_MASS_BOUND,
    "lower_R_B_bound": LOWER_R_B_bound,
    "upper_R_B_bound": UPPER_R_B_bound,
    "f_SN_1_min": F_SN_1_MIN,
}

PRIOR_KEY, RNG_KEY = random.split(RNG_KEY)
prior_predective = Predictive(MODEL, num_samples=10000)
prior_predictions = prior_predective(PRIOR_KEY, **prior_inputs)

prior_mb = prior_predictions["sn_observables"][:, :, 0]
prior_x1 = prior_predictions["sn_observables"][:, :, 1]
prior_c_app = prior_predictions["sn_observables"][:, :, 2]

print("\nPlotting prior predictive...\n")

prior_mb_low, prior_mb_median, prior_mb_high = jnp.percentile(
    prior_mb, jnp.array([16, 50, 84]), axis=0
)
prior_x1_low, prior_x1_median, prior_x1_high = jnp.percentile(
    prior_x1, jnp.array([16, 50, 84]), axis=0
)
prior_c_app_low, prior_c_app_median, prior_c_app_high = jnp.percentile(
    prior_c_app, jnp.array([16, 50, 84]), axis=0
)

ncols = 3
if "Mass" in MODEL_NAME:
    ncols = 4
    prior_mass = prior_predictions["host_observables"]
    prior_mass_low, prior_mass_median, prior_mass_high = jnp.percentile(
        prior_mass, jnp.array([16, 50, 84]), axis=0
    )

fig, ax = plt.subplots(ncols=ncols, figsize=(16, 6))
ax[0].errorbar(
    sn_observables[:, 0],
    prior_mb_median,
    yerr=[prior_mb_median - prior_mb_low, prior_mb_high - prior_mb_median],
    fmt=".",
    color=default_colors[0],
)
ax[0].set_xlabel(r"$m_{B,\mathrm{obs}}$", fontsize=25)
ax[0].set_ylabel(r"$m_{B,\mathrm{prior}}$", fontsize=25)

ax[1].errorbar(
    sn_observables[:, 1],
    prior_x1_median,
    yerr=[prior_x1_median - prior_x1_low, prior_x1_high - prior_x1_median],
    fmt=".",
    color=default_colors[0],
)
ax[1].set_xlabel(r"$x_{1,\mathrm{obs}}$", fontsize=25)
ax[1].set_ylabel(r"$x_{1,\mathrm{prior}}$", fontsize=25)

ax[2].errorbar(
    sn_observables[:, 2],
    prior_c_app_median,
    yerr=[prior_c_app_median - prior_c_app_low, prior_c_app_high - prior_c_app_median],
    fmt=".",
    color=default_colors[0],
)
ax[2].set_xlabel(r"$c_{\mathrm{obs}}$", fontsize=25)
ax[2].set_ylabel(r"$c_{\mathrm{prior}}$", fontsize=25)

if "Mass" in MODEL_NAME:
    ax[3].errorbar(
        host_observables,
        prior_mass_median,
        yerr=[prior_mass_median - prior_mass_low, prior_mass_high - prior_mass_median],
        fmt=".",
        color=default_colors[0],
    )
    ax[3].set_xlabel(r"$M_{\mathrm{host}}$", fontsize=25)
    ax[3].set_ylabel(r"$M_{\mathrm{host,prior}}$", fontsize=25)

fig.tight_layout()
fig.savefig(
    fig_path / ("prior_observables.png"),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)

print("\nPlotting prior GTC...\n")

(
    filtered_samples,
    pop_1_samples,
    pop_2_samples,
    param_labels,
    pop_labels,
) = preprocess_samples(prior_predictions, shared_params)

corner(
    chains=[filtered_samples],
    param_labels=param_labels,
    chain_labels=["Prior"],
    make_transparent=True,
    save_path=fig_path / "prior_gtc_delta.png",
)

corner(
    chains=[pop_2_samples, pop_1_samples],
    param_labels=pop_labels,
    chain_labels=["Population 2", "Population 1"],
    make_transparent=True,
    save_path=fig_path / "prior_gtc_pops.png",
)

# -------------------------- RUN MCMC --------------------------------- #

print("\nFine-tuning MCLMC...\n")

cosmology = jc.Planck15()

model_inputs = {
    "sn_observables": sn_observables,
    "sn_covariances": sn_covariances,
    "sn_redshifts": sn_redshifts,
    "host_mass": host_observables,
    "host_mass_err": host_uncertainty,
    "host_mean": host_mean,
    "host_std": host_std,
    "cosmology": cosmology,
    "verbose": VERBOSE,
    "lower_host_mass_bound": LOWER_HOST_MASS_BOUND,
    "upper_host_mass_bound": UPPER_HOST_MASS_BOUND,
    "lower_R_B_bound": LOWER_R_B_bound,
    "upper_R_B_bound": UPPER_R_B_bound,
    "f_SN_1_min": F_SN_1_MIN,
}

# MODEL_KEY, INIT_KEY, TUNE_KEY, RNG_KEY = random.split(RNG_KEY, 4)

# init_params, potential_fn_gen, transform_fn_gen, _ = initialize_model(
#     MODEL_KEY,
#     MODEL,
#     model_kwargs=model_inputs,
#     dynamic_args=True,
#     init_strategy=init_to_sample,
#     # forward_mode_differentiation=True,
# )

# logdensity_fn = lambda position: -potential_fn_gen(**model_inputs)(position)
# transform_fn = lambda position: transform_fn_gen(**model_inputs)(position)
# initial_position = init_params.z

# initial_state = blackjax.mcmc.mclmc.init(
#     position=initial_position, logdensity_fn=logdensity_fn, rng_key=INIT_KEY
# )

# kernel = blackjax.mcmc.mclmc.build_kernel(
#     logdensity_fn=logdensity_fn,
#     integrator=blackjax.mcmc.integrators.noneuclidean_mclachlan,
# )

# (
#     blackjax_state_after_tuning,
#     blackjax_mclmc_sampler_params,
# ) = blackjax.mclmc_find_L_and_step_size(
#     mclmc_kernel=kernel,
#     num_steps=NUM_ADAPTATION,
#     state=initial_state,
#     rng_key=TUNE_KEY,
# )

# print(f"\nTuning Steps: {NUM_ADAPTATION}")
# print(f"Found L = {blackjax_mclmc_sampler_params.L}")
# print(f"Found step size = {blackjax_mclmc_sampler_params.step_size}\n")

# sampling_alg = blackjax.mclmc(
#     logdensity_fn,
#     L=blackjax_mclmc_sampler_params.L,
#     step_size=blackjax_mclmc_sampler_params.step_size,
# )


# print("\nRunning MCMC...\n")

# initial_state = blackjax_state_after_tuning
# posterior_samples = {}

# for i in range(len(SAMPLE_STEPS)):
#     RUN_KEY, RNG_KEY = random.split(RNG_KEY)
#     n_steps = SAMPLE_STEPS[i]
#     cumulative_n_steps = jnp.sum(SAMPLE_STEPS[: i + 1])
#     print(f"\nSampling Steps: {cumulative_n_steps}/{NUM_SAMPLES}")

#     initial_state, state_history, info_history = run_inference_algorithm(
#         rng_key=RUN_KEY,
#         initial_state=initial_state,
#         inference_algorithm=sampling_alg,
#         num_steps=n_steps,
#         progress_bar=True,
#         # transform=transform_fn,
#     )

#     transformed_state = jax.vmap(transform_fn)(state_history.position)

#     if i == 0:
#         for key in shared_params + independent_params + ["f_SN_1_mid"]:
#             if key in transformed_state.keys():
#                 posterior_samples[key] = transformed_state[key]
#     else:
#         for key in posterior_samples.keys():
#             posterior_samples[key] = jnp.concatenate(
#                 (posterior_samples[key], transformed_state[key]), axis=0
#             )

SAMPLING_KEY, RNG_KEY = random.split(RNG_KEY)

(
    final_state,
    state_history,
    info_history,
    transformed_state,
) = sampling.run_mclcm_sampler(
    rng_key=RNG_KEY,
    model=MODEL,
    model_kwargs=model_inputs,
    num_samples=NUM_SAMPLES,
    num_tuning_steps=NUM_ADAPTATION,
    num_chains=NUM_CHAINS,
    verbose=True,
)

seeded_model = seed(MODEL, RNG_KEY)
model_trace = trace(seeded_model).get_trace(**model_inputs)
sample_keys = [
    key
    for key in model_trace.keys()
    if model_trace[key]["type"] == "sample" and not model_trace[key]["is_observed"]
]
samples = {
    sample_key: transformed_state[sample_key]
    for sample_key in sample_keys
    if sample_key in transformed_state.keys()
}
samples_array = []
for key in samples.keys():
    flattened_sample = samples[key]
    shape = flattened_sample.shape
    if len(shape) == 1:
        samples_array.append(flattened_sample[:, None, None])
    elif len(shape) == 2:
        samples_array.append(flattened_sample[..., None])
    else:
        samples_array.append(flattened_sample.reshape(NUM_CHAINS, NUM_SAMPLES, -1))
samples_array = np.concatenate(samples_array, axis=-1)

autocorr_steps = np.exp(np.linspace(np.log(100), np.log(NUM_SAMPLES), 10)).astype(int)
taus = np.array(
    [sampling.autocorrelation_time(samples_array[:, :n, :]) for n in autocorr_steps]
)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(autocorr_steps, taus, "o-", color=default_colors[0])
ylim = ax.get_ylim()
ax.plot(autocorr_steps, autocorr_steps / 50, "--", color="k")
ax.set_ylim(ylim)
ax.set_xlabel("Number of Steps", fontsize=25)
ax.set_ylabel(r"$\tau$", fontsize=25)
fig.tight_layout()
fig.savefig(
    fig_path / ("autocorrelation" + RUN_MODIFIER + ".png"),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)

av_trace = sampling.arviz_from_states(samples, info_history)

with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    1000,
):
    summary = az.summary(av_trace)
    print(summary)

posterior_samples = {
    key: sampling.flatten_parameter(transformed_state[key])
    for key in shared_params + ["f_SN_1_mid"]
    if key in transformed_state.keys()
}


print("\n\nPlotting Posterior GTC...\n")

(
    filtered_samples,
    pop_1_samples,
    pop_2_samples,
    param_labels,
    pop_labels,
) = preprocess_samples(posterior_samples, shared_params)

corner(
    chains=[filtered_samples],
    param_labels=param_labels,
    chain_labels=["Posterior"],
    make_transparent=True,
    save_path=fig_path / "posterior_gtc_delta.png",
)

corner(
    chains=[pop_2_samples, pop_1_samples],
    param_labels=pop_labels,
    chain_labels=["Population 2", "Population 1"],
    make_transparent=True,
    save_path=fig_path / "posterior_gtc_pops.png",
)

# -------------------------- PLOT POP FRACTION --------------------------------- #

print("\nPlotting population fraction...\n")

if MODEL_NAME == "SNMass" or MODEL_NAME == "SN2PopMass" or MODEL_NAME == "SNMassDelta":
    idx = np.random.choice(
        len(posterior_samples["f_SN_1_mid"]), size=MAX_SAMPLES * 5, replace=False
    )
    f_SN_1_mid = posterior_samples["f_SN_1_mid"][idx]
    scaling = posterior_samples["scaling"][idx]
    if MODEL_NAME == "SNMass" or MODEL_NAME == "SNMassDelta":
        mean_mass = posterior_samples["M_host"][idx]
        mass_scatter = posterior_samples["M_host_scatter"][idx]
    elif MODEL_NAME == "SN2PopMass":
        mean_mass = posterior_samples["M_host_mixture_mean"]
        mass_scatter = posterior_samples["M_host_mixture_std"]

    mass = np.linspace(6, 13, 100)
    f_1_samples = np.zeros((MAX_SAMPLES, 100))
    for i in range(MAX_SAMPLES):
        obs_rescaled_mass = (mass - host_mean) / host_std
        rescaled_mass = (obs_rescaled_mass - mean_mass[i]) / mass_scatter[i]
        linear_function = scaling[i] * rescaled_mass
        f_1 = sigmoid(x=linear_function, f_mid=f_SN_1_mid[i], f_min=F_SN_1_MIN)
        f_1_samples[i] = f_1

    f_1_percentiles = np.percentile(f_1_samples, [16, 50, 84], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(mass, f_1_percentiles[1], color=default_colors[0], lw=3)
    ax.fill_between(
        mass, f_1_percentiles[0], f_1_percentiles[2], color=default_colors[0], alpha=0.3
    )
    ax.set_xlabel(r"$\log_{10}(M_{\mathrm{host}})$", fontsize=25)
    ax.set_ylabel(r"$f^{\mathrm{SN}}_1$", fontsize=25)
    fig.tight_layout()
    fig.savefig(
        fig_path / ("sn_fraction" + RUN_MODIFIER + ".png"),
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

if MODEL_NAME == "SNMassGP":
    mass = posterior_samples["eval_mass"][0]
    gp_low, gp_median, gp_high = jnp.percentile(
        posterior_samples["gp_eval"], jnp.array([16, 50, 84]), axis=0
    )
    f_sn_1_low, f_sn_1_median, f_sn_1_high = jnp.percentile(
        posterior_samples["f_sn_1_eval"], jnp.array([16, 50, 84]), axis=0
    )

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].plot(mass, gp_median, color=default_colors[0], lw=3)
    ax[0].fill_between(
        mass, gp_low, gp_high, color=default_colors[0], alpha=0.3, label="GP"
    )
    ax[0].set_xlabel(r"$\log_{10}(M_{\mathrm{host}})$", fontsize=25)
    ax[0].set_ylabel(r"$\mathrm{GP}(M_{\mathrm{host}})$", fontsize=25)

    ax[1].plot(mass, f_sn_1_median, color=default_colors[0], lw=3)
    ax[1].fill_between(
        mass,
        f_sn_1_low,
        f_sn_1_high,
        color=default_colors[0],
        alpha=0.3,
        label=r"$f^{\mathrm{SN}}_1$",
    )
    ax[1].set_xlabel(r"$\log_{10}(M_{\mathrm{host}})$", fontsize=25)
    ax[1].set_ylabel(r"$f^{\mathrm{SN}}_1$", fontsize=25)

    fig.tight_layout()
    fig.savefig(
        fig_path / ("sn_fraction" + RUN_MODIFIER + ".png"),
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
