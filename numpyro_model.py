# -------------------------- IMPORTS --------------------------------- #

import numpyro as npy
npy.util.enable_x64()
npy.util.set_platform("cpu")

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

from jax import random
from pathlib import Path
from numpyro.infer import MCMC, NUTS, Predictive
from bayesnova.numpyro_models import SN, SNMass, SN2PopMass, run_mcmc

default_colors = sns.color_palette("colorblind")

COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['axes.edgecolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

# -------------------------- HELPER FUNCTIONS --------------------------------- #

def map_to_latex(label: str):

    map_dict = {
        'H0': r'$H_0$',
        'f_sn_1': r'$f_1^{\mathrm{SN}}$',
        'f_1_max': r'$f_{1,\mathrm{max}}^{\mathrm{SN}}$',
        'M_int': r'$M_{\mathrm{int}}$',
        'M_int_scatter': r'$\sigma_{M_{\mathrm{int}}}$',
        'alpha': r'$\hat{\alpha}$',
        'X_int': r'$\hat{X}_{\mathrm{1,int}}$',
        'X_int_scatter': r'$\sigma_{X_{\mathrm{1,int}}}$',
        'beta': r'$\hat{\beta}$',
        'C_int': r'$\hat{c}_{\mathrm{int}}$',
        'C_int_scatter': r'$\sigma_{c_{\mathrm{int}}}$',
        'R_B': r'$\hat{R}_B$',
        'R_B_scatter': r'$\sigma_{R_B}$',
        'gamma_EBV': r'$\gamma_{\mathrm{E(B-V)}}$',
        'tau_EBV': r'$\tau_{\mathrm{E(B-V)}}$',
        'M_host': r'$\hat{M}_{\mathrm{host}}$',
        'M_host_scatter': r'$\sigma_{M_{\mathrm{host}}}$',
        'scaling': r'$a$',
        'offset': r'$b$',
    }

    return map_dict[label]

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

# -------------------------- SETTINGS --------------------------------- #

NUM_WARMUP = 10000#10000
NUM_SAMPLES = 1000#75000
NUM_CHAINS = 1

RUN_NAME = "Supercal_Hubble_Flow"
MODEL_NAME = "SN"
MODEL = globals()[MODEL_NAME]

print("\nModel: ", MODEL_NAME)
print("Run Name: ", RUN_NAME)
print("Num Warmup: ", NUM_WARMUP)
print("Num Samples: ", NUM_SAMPLES)
print("Num Chains: ", NUM_CHAINS, "\n")

base_path = Path("/groups/dark/osman/BayeSNova/")
data_path = base_path / "data"

output_path = base_path / "output" / MODEL_NAME / RUN_NAME
output_path.mkdir(parents=True, exist_ok=True)
fig_path = output_path / "figures"
fig_path.mkdir(parents=True, exist_ok=True)
posterior_path = output_path / "posteriors"
posterior_path.mkdir(parents=True, exist_ok=True)

cfg_path = data_path / "config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg['model_cfg']['host_galaxy_cfg']['use_properties'] = True
cfg['model_cfg']['host_galaxy_cfg']['independent_property_names'] = ['global_mass']

# -------------------------- LOAD DATA --------------------------------- #

print("\nLoading data...\n")

data = pd.read_csv(data_path / "supercal_hubble_flow.dat", sep=" ")

prep.init_global_data(data, None, cfg['model_cfg'])

host_observables = prep.host_galaxy_observables.squeeze()
host_covariances = prep.host_galaxy_covariances.squeeze()

idx_valid_mass = host_observables != -9999.
idx_not_calibrator = ~prep.idx_calibrator_sn
idx_valid = idx_not_calibrator

if MODEL_NAME == "SNMass":
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

# -------------------------- RUN MCMC --------------------------------- #

print("\nRunning MCMC...\n")

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
rng_key, subkey = random.split(subkey)

mcmc = run_mcmc(
    model=MODEL,
    rng_key=rng_key,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    model_kwargs=model_inputs,
)


posterior_samples = mcmc.get_samples()

# Pickle the posterior_samples object
with open(posterior_path / "posterior_samples.pkl", "wb") as file:
   dill.dump(posterior_samples, file)

with open(posterior_path / "mcmc.pkl", "wb") as file:
    dill.dump(mcmc, file)

with open(posterior_path / "mcmc.pkl", "rb") as file:
    mcmc = dill.load(file)

mcmc.print_summary()

# -------------------------- PLOT TRIANGLE --------------------------------- #

print("\nPlotting triangle...\n")

shared_params = [
    'alpha', 'beta', 'R_B', 'R_B_scatter', 'gamma_EBV',
    'M_host', 'M_host_scatter', 'scaling', 'offset', 'f_1_max' 
]
independent_params = [
    'M_int', 'X_int', 'X_int_scatter', 'C_int', 'C_int_scatter', 'tau_EBV'
]

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
    param_posterior_samples.append(posterior_samples[param][:,0])
    param_posterior_samples.append(posterior_samples[param][:,1])
    param_labels.append(map_to_latex(param))
    idx_counter += 2

param_posterior_samples = np.array(param_posterior_samples).T

params_to_skip = [i for i in range(n_cosmo_and_shared)]
pop_1_idx = np.array(
    params_to_skip + pop_1_idx
)
pop_2_idx = np.array(
    params_to_skip + pop_2_idx
)

pop_1_samples = param_posterior_samples[:, pop_1_idx]
pop_2_samples = param_posterior_samples[:, pop_2_idx]

chain_labels = ['Population 2', 'Population 1']

print("Pop 1 Samples Shape: ", pop_1_samples.shape)
print("Pop 2 Samples Shape: ", pop_2_samples.shape)
print("No. of labels: ", len(param_labels))

GTC = pygtc.plotGTC(
    chains=[pop_2_samples, pop_1_samples],
    paramNames=param_labels,
    chainLabels=chain_labels,
    nContourLevels=3,
    legendMarker='All',
    customLabelFont={'family':'Arial', 'size':15},
    customTickFont={'family':'Arial', 'size':6},
);

for axes in GTC.axes:
    for spine in axes.spines:
        axes.spines[spine].set_color("white")
    axes.tick_params(axis='both', colors='white')

GTC.savefig(fig_path / "gtc.png", transparent=True, dpi=300)

# -------------------------- PLOT POP FRACTION --------------------------------- #

print("\nPlotting population fraction...\n")

if MODEL_NAME == "SNMass":

    f_1_max = posterior_samples["f_1_max"]
    scaling = posterior_samples["scaling"]
    offset = posterior_samples["offset"]
    mean_mass = posterior_samples["M_host"]

    mass = np.linspace(6, 13, 100)
    f_1_samples = np.zeros((NUM_SAMPLES, 100))
    for i in range(NUM_SAMPLES):
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