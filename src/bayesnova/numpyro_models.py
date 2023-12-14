import jax
import yaml
import pickle
import numpy as np
import pandas as pd
import numpyro as npy

npy.util.enable_x64()
import jax_cosmo as jc
import jax.numpy as jnp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpyro.distributions as dist

# import bayesnova.old_src.preprocessing as prep

from jax import random
from pathlib import Path
from numpyro.infer import MCMC, NUTS, SVI, TraceEnum_ELBO, Predictive
from astropy.cosmology import Planck15
from numpyro.handlers import seed, trace, reparam
from numpyro.infer.reparam import LocScaleReparam
from numpyro.contrib.funsor import config_enumerate
from numpyro.distributions.transforms import OrderedTransform
from numpyro.ops.indexing import Vindex
from tinygp import kernels, GaussianProcess


# ------------------- HELPER FUNCTIONS ------------------- #


def sigmoid(x, slope=1, midpoint=0.0, f_mid=0.5, f_min=0.0):
    numerator = 2 * f_mid - f_min
    denominator = 1 + jnp.exp(-slope * (x - midpoint))

    output = numerator / denominator + f_min

    return output


def distance_moduli(cosmology, redshifts):
    scale_factors = jc.utils.z2a(redshifts)
    dA = jc.background.angular_diameter_distance(cosmology, scale_factors) / cosmology.h
    dL = (1.0 + redshifts) ** 2 * dA
    mu = 5.0 * jnp.log10(jnp.abs(dL)) + 25.0

    return mu


def delta_mass_step(mass, mass_step, mass_cutoff=10.0):
    return mass_step * (1.0 / (1.0 + jnp.exp((mass - mass_cutoff) / 0.01)) - 0.5)


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


def run_mcmc(
    model, rng_key, num_warmup=500, num_samples=1000, num_chains=1, model_kwargs={}
):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    mcmc.run(rng_key=rng_key, **model_kwargs)

    return mcmc


# ------------------- BASIC MODEL DEFINITIONS ------------------- #


def SNTripp(
    sn_observables=None,
    sn_covariances=None,
    sn_redshifts=None,
    host_mass=None,
    host_mass_err=None,
    cosmology=None,
    constraint_factor=-500,
    verbose=False,
    *args,
    **kwargs,
):
    M_int = npy.sample("M_int", dist.Normal(-19.5, 1.0))
    M_int_scatter = npy.sample("M_int_scatter", dist.HalfNormal(3.0))
    alpha = npy.sample("alpha", dist.Normal(0, 0.1))
    beta = npy.sample("beta", dist.Normal(0, 1))

    if verbose:
        print("\n---------------- MODEL SHAPES ----------------")
        print(f"M_int.shape: {M_int.shape}")
        print(f"M_int_scatter.shape: {M_int_scatter.shape}")
        print(f"alpha.shape: {alpha.shape}")
        print(f"beta.shape: {beta.shape}\n")

    with npy.plate("sn", size=len(sn_redshifts)):
        vindexed_cov = Vindex(sn_covariances)
        distance_modulus = distance_moduli(cosmology, sn_redshifts)
        app_mag = (
            M_int
            + distance_modulus
            - alpha * sn_observables[..., 1]
            + beta * sn_observables[..., 2]
        )

        if verbose:
            print(f"distance_modulus.shape: {distance_modulus.shape}")
            print(f"app_mag.shape: {app_mag.shape}\n")

        app_mag_err = jnp.sqrt(
            vindexed_cov[..., 0, 0]
            + alpha**2 * vindexed_cov[..., 1, 1]
            + beta**2 * vindexed_cov[..., 2, 2]
            + 2 * alpha * vindexed_cov[..., 0, 1]
            - 2 * beta * vindexed_cov[..., 0, 2]
            - 2 * alpha * beta * vindexed_cov[..., 1, 2]
            + M_int_scatter**2
        )

        if verbose:
            print(f"app_mag_err.shape: {app_mag_err.shape}\n")

        npy.sample(
            "app_mag", dist.Normal(app_mag, app_mag_err), obs=sn_observables[..., 0]
        )


def SNTrippMass(
    sn_observables=None,
    sn_covariances=None,
    sn_redshifts=None,
    host_mass=None,
    host_mass_err=None,
    cosmology=None,
    constraint_factor=-500,
    mass_cutoff=10.0,
    verbose=False,
    *args,
    **kwargs,
):
    M_int = npy.sample("M_int", dist.Normal(-19.5, 1.0))
    M_int_scatter = npy.sample("M_int_scatter", dist.HalfNormal(3.0))
    alpha = npy.sample("alpha", dist.Normal(0, 0.1))
    beta = npy.sample("beta", dist.Normal(0, 1))
    mass_step = npy.sample("mass_step", dist.Normal(0, 1))

    if verbose:
        print("\n---------------- MODEL SHAPES ----------------")
        print(f"M_int.shape: {M_int.shape}")
        print(f"M_int_scatter.shape: {M_int_scatter.shape}")
        print(f"alpha.shape: {alpha.shape}")
        print(f"beta.shape: {beta.shape}")
        print(f"mass_step.shape: {mass_step.shape}\n")

    with npy.plate("sn", size=len(sn_redshifts)):
        vindexed_cov = Vindex(sn_covariances)
        distance_modulus = distance_moduli(cosmology, sn_redshifts)
        delta_mag = delta_mass_step(host_mass, mass_step, mass_cutoff=mass_cutoff)
        app_mag = (
            M_int
            + distance_modulus
            - alpha * sn_observables[..., 1]
            + beta * sn_observables[..., 2]
            + delta_mag
        )

        if verbose:
            print(f"distance_modulus.shape: {distance_modulus.shape}")
            print(f"app_mag.shape: {app_mag.shape}\n")

        app_mag_err = jnp.sqrt(
            vindexed_cov[..., 0, 0]
            + alpha**2 * vindexed_cov[..., 1, 1]
            + beta**2 * vindexed_cov[..., 2, 2]
            + 2 * alpha * vindexed_cov[..., 0, 1]
            - 2 * beta * vindexed_cov[..., 0, 2]
            - 2 * alpha * beta * vindexed_cov[..., 1, 2]
            + M_int_scatter**2
        )

        if verbose:
            print(f"app_mag_err.shape: {app_mag_err.shape}\n")

        npy.sample(
            "app_mag", dist.Normal(app_mag, app_mag_err), obs=sn_observables[..., 0]
        )


# ------------------- HIERARCHICAL MODEL DEFINITIONS ------------------- #

SN_reparam_cfg = {
    "X_int_latent": LocScaleReparam(0.0),
    "C_int_latent": LocScaleReparam(0.0),
    "R_B_latent": LocScaleReparam(0.0),
}


@config_enumerate
@reparam(config=SN_reparam_cfg)
def SN(
    sn_observables=None,
    sn_covariances=None,
    sn_redshifts=None,
    host_mass=None,
    host_mass_err=None,
    cosmology=None,
    constraint_factor=-500,
    verbose=False,
    *args,
    **kwargs,
):
    # sn_observables: [N, 3] array of observables
    # sn_covariance: [N, 3, 3] array of covariance matrices
    # sn_redshifts: [N] array of redshifts
    # host_mass: [N] array of host masses

    # Define Cosmology
    cosmo = cosmology

    # Define priors on independent population parameters
    if verbose:
        print("\n---------------- MODEL SHAPES ----------------")

    X_int = npy.sample(
        "X_int",
        dist.TransformedDistribution(
            dist.Normal(jnp.array([0.0, 0.0]), jnp.array([5.0, 5.0])),
            OrderedTransform(),
        ),
    )

    if verbose:
        print(f"X_int.shape: {X_int.shape}")

    with npy.plate("sn_populations", size=2):
        M_int = npy.sample(
            "M_int", dist.TruncatedNormal(-19.5, 1.0, high=-15.0, low=-25.0)
        )
        X_int_scatter = npy.sample("X_int_scatter", dist.HalfNormal(5.0))
        C_int = npy.sample("C_int", dist.Normal(0.0, 5.0))
        C_int_scatter = npy.sample("C_int_scatter", dist.HalfNormal(5.0))
        tau_EBV = npy.sample("tau_EBV", dist.HalfNormal(5.0))

    if verbose:
        print(f"M_int.shape: {M_int.shape}")
        print(f"X_int_scatter.shape: {X_int_scatter.shape}")
        print(f"C_int.shape: {C_int.shape}")
        print(f"C_int_scatter.shape: {C_int_scatter.shape}")
        print(f"tau_EBV.shape: {tau_EBV.shape}\n")

    # Define priors on shared population parameters

    alpha = npy.sample("alpha", dist.Normal(0, 5.0))
    beta = npy.sample("beta", dist.Normal(0, 5.0))
    R_B = npy.sample("R_B", dist.HalfNormal(5.0))
    R_B_scatter = npy.sample("R_B_scatter", dist.HalfNormal(5.0))
    unshifted_gamma_EBV = npy.sample("unshifted_gamma_EBV", dist.HalfNormal(5.0))
    gamma_EBV = npy.deterministic("gamma_EBV", unshifted_gamma_EBV + 1)

    if verbose:
        print(f"alpha.shape: {alpha.shape}")
        print(f"beta.shape: {beta.shape}")
        print(f"R_B.shape: {R_B.shape}")
        print(f"R_B_scatter.shape: {R_B_scatter.shape}")
        print(f"unshifted_gamma_EBV.shape: {unshifted_gamma_EBV.shape}")
        print(f"gamma_EBV.shape: {gamma_EBV.shape}")

    # Population fractions
    f_sn_1 = npy.sample("f_sn_1", dist.Uniform(0.15, 0.85))
    f_sn = jnp.array([f_sn_1, 1 - f_sn_1])

    if verbose:
        print(f"f_sn_1.shape: {f_sn_1.shape}")
        print(f"f_sn.shape: {f_sn.shape}\n")

    with npy.plate("sn", size=len(sn_redshifts)):
        # SN Population assignment

        population_assignment = npy.sample(
            "population_assignment",
            dist.Categorical(f_sn),
            infer={"enumerate": "parallel"},
        )
        M_int_latent = Vindex(M_int)[..., population_assignment]

        if verbose:
            print(f"population_assignment.shape: {population_assignment.shape}")
            print(f"M_int_latent.shape: {M_int_latent.shape}\n")

        X_int_latent_SN_pop_1 = npy.sample(
            "X_int_latent_SN_pop_1",
            dist.Normal(Vindex(X_int)[..., 0], Vindex(X_int_scatter)[..., 0]),
        )
        X_int_latent_SN_pop_2 = npy.sample(
            "X_int_latent_SN_pop_2",
            dist.Normal(Vindex(X_int)[..., 1], Vindex(X_int_scatter)[..., 1]),
        )
        X_int_latent_values = jnp.stack(
            [X_int_latent_SN_pop_1, X_int_latent_SN_pop_2], axis=-1
        )
        X_int_latent = npy.deterministic(
            "X_int_latent", Vindex(X_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"X_int_latent_SN_pop_1.shape: {X_int_latent_SN_pop_1.shape}")
            print(f"X_int_latent_SN_pop_2.shape: {X_int_latent_SN_pop_2.shape}")
            print(f"X_int_latent_values.shape: {X_int_latent_values.shape}")
            print(f"X_int_latent.shape: {X_int_latent.shape}\n")

        C_int_latent_SN_pop_1 = npy.sample(
            "C_int_latent_SN_pop_1",
            dist.Normal(Vindex(C_int)[..., 0], Vindex(C_int_scatter)[..., 0]),
        )
        C_int_latent_SN_pop_2 = npy.sample(
            "C_int_latent_SN_pop_2",
            dist.Normal(Vindex(C_int)[..., 1], Vindex(C_int_scatter)[..., 1]),
        )
        C_int_latent_values = jnp.stack(
            [C_int_latent_SN_pop_1, C_int_latent_SN_pop_2], axis=-1
        )
        C_int_latent = npy.deterministic(
            "C_int_latent", Vindex(C_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"C_int_latent_SN_pop_1.shape: {C_int_latent_SN_pop_1.shape}")
            print(f"C_int_latent_SN_pop_2.shape: {C_int_latent_SN_pop_2.shape}")
            print(f"C_int_latent_values.shape: {C_int_latent_values.shape}")
            print(f"C_int_latent.shape: {C_int_latent.shape}\n")

        EBV_latent_decentered = npy.sample(
            "EBV_latent_decentered", dist.Gamma(gamma_EBV)
        )
        EBV_latent_SN_pop_1 = npy.deterministic(
            "EBV_latent_SN_pop_1", EBV_latent_decentered * Vindex(tau_EBV)[..., 0]
        )
        EBV_latent_SN_pop_2 = npy.deterministic(
            "EBV_latent_SN_pop_2", EBV_latent_decentered * Vindex(tau_EBV)[..., 1]
        )
        EBV_latent_values = jnp.stack(
            [EBV_latent_SN_pop_1, EBV_latent_SN_pop_2], axis=-1
        )
        EBV_latent = npy.deterministic(
            "EBV_latent", Vindex(EBV_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"EBV_latent_decentered.shape: {EBV_latent_decentered.shape}")
            print(f"EBV_latent_SN_pop_1.shape: {EBV_latent_SN_pop_1.shape}")
            print(f"EBV_latent_SN_pop_2.shape: {EBV_latent_SN_pop_2.shape}")
            print(f"EBV_latent_values.shape: {EBV_latent_values.shape}")
            print(f"EBV_latent.shape: {EBV_latent.shape}\n")

        R_B_latent = npy.sample(
            "R_B_latent",
            dist.Normal(
                loc=R_B,
                scale=R_B_scatter,  # low=1,
            ),
        )

        if verbose:
            print(f"R_B_latent.shape: {R_B_latent.shape}\n")

        distance_modulus = distance_moduli(cosmo, sn_redshifts)

        if verbose:
            print(f"distance_modulus.shape: {distance_modulus.shape}\n")

        app_mag_1 = (
            Vindex(M_int)[..., 0]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_1
            + beta * C_int_latent_SN_pop_1
            + R_B_latent * EBV_latent_SN_pop_1
        )
        app_stretch_1 = X_int_latent_SN_pop_1
        app_color_1 = C_int_latent_SN_pop_1 + EBV_latent_SN_pop_1
        mean_1 = jnp.stack([app_mag_1, app_stretch_1, app_color_1], axis=-1)
        sn_dist_1 = dist.MultivariateNormal(mean_1, sn_covariances)

        if verbose:
            print(f"app_mag_1.shape: {app_mag_1.shape}")
            print(f"app_stretch_1.shape: {app_stretch_1.shape}")
            print(f"app_color_1.shape: {app_color_1.shape}")
            print(f"mean_1.shape: {mean_1.shape}")
            print(f"sn_dist_1.shape: {sn_dist_1.shape()}\n")

        app_mag_2 = (
            Vindex(M_int)[..., 1]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_2
            + beta * C_int_latent_SN_pop_2
            + R_B_latent * EBV_latent_SN_pop_2
        )
        app_stretch_2 = X_int_latent_SN_pop_2
        app_color_2 = C_int_latent_SN_pop_2 + EBV_latent_SN_pop_2
        mean_2 = jnp.stack([app_mag_2, app_stretch_2, app_color_2], axis=-1)
        sn_dist_2 = dist.MultivariateNormal(mean_2, sn_covariances)

        if verbose:
            print(f"app_mag_2.shape: {app_mag_2.shape}")
            print(f"app_stretch_2.shape: {app_stretch_2.shape}")
            print(f"app_color_2.shape: {app_color_2.shape}")
            print(f"mean_2.shape: {mean_2.shape}")
            print(f"sn_dist_2.shape: {sn_dist_2.shape()}\n")

        apparent_magnitude_values = jnp.stack([app_mag_1, app_mag_2], axis=-1).squeeze()

        if verbose:
            print(f"apparent_magnitude_values.shape: {apparent_magnitude_values.shape}")

        if sn_observables is not None:
            sn_log_prob_1 = sn_dist_1.log_prob(sn_observables) + jnp.log(f_sn_1)
            sn_log_prob_2 = sn_dist_2.log_prob(sn_observables) + jnp.log(1 - f_sn_1)
            npy.deterministic(
                "sn_log_membership_ratio", (sn_log_prob_1 - sn_log_prob_2) / jnp.log(10)
            )

        apparent_magnitude = npy.deterministic(
            "apparent_magnitude",
            Vindex(apparent_magnitude_values)[..., population_assignment],
        )
        apparent_stretch = npy.deterministic("apparent_stretch", X_int_latent)
        apparent_color = npy.deterministic("apparent_color", C_int_latent + EBV_latent)

        if verbose:
            print(f"apparent_magnitude.shape: {apparent_magnitude.shape}")
            print(f"apparent_stretch.shape: {apparent_stretch.shape}")
            print(f"apparent_color.shape: {apparent_color.shape}")

        # Sample observables
        mean = jnp.stack(
            [apparent_magnitude, apparent_stretch, apparent_color], axis=-1
        )

        if verbose:
            print(f"mean.shape: {mean.shape}\n")
            print(f"sn_observables.shape: {sn_observables.shape}")
            print(f"sn_covariances.shape: {sn_covariances.shape}\n")

        result_dist = dist.MultivariateNormal(mean, sn_covariances)
        npy.sample("sn_observables", result_dist, obs=sn_observables)


SNMass_reparam_cfg = {
    "X_int_latent": LocScaleReparam(0.0),
    "C_int_latent": LocScaleReparam(0.0),
    "R_B_latent": LocScaleReparam(0.0),
    "M_host_latent": LocScaleReparam(0.0),
}


@config_enumerate
@reparam(config=SNMass_reparam_cfg)
def SNMass(
    sn_observables=None,
    sn_covariances=None,
    sn_redshifts=None,
    host_mass=None,
    host_mass_err=None,
    cosmology=None,
    constraint_factor=-500,
    verbose=False,
    host_mean=0.0,
    host_std=1.0,
    f_SN_1_min=0.15,
    *args,
    **kwargs,
):
    # sn_observables: [N, 3] array of observables
    # sn_covariance: [N, 3, 3] array of covariance matrices
    # sn_redshifts: [N] array of redshifts
    # host_mass: [N] array of host masses

    # Define Cosmology
    cosmo = cosmology

    # Define priors on independent population parameters

    if verbose:
        print("\n---------------- MODEL SHAPES ----------------")

    X_int = npy.sample(
        "X_int",
        dist.TransformedDistribution(
            dist.Normal(jnp.array([0.0, 0.0]), jnp.array([5.0, 5.0])),
            OrderedTransform(),
        ),
    )

    if verbose:
        print(f"X_int.shape: {X_int.shape}")

    with npy.plate("sn_populations", size=2):
        M_int = npy.sample(
            "M_int", dist.TruncatedNormal(-19.5, 1, high=-15.0, low=-25.0)
        )
        X_int_scatter = npy.sample("X_int_scatter", dist.HalfNormal(5.0))
        C_int = npy.sample("C_int", dist.Normal(0.0, 5.0))
        C_int_scatter = npy.sample("C_int_scatter", dist.HalfNormal(5.0))
        tau_EBV = npy.sample("tau_EBV", dist.HalfNormal(5.0))

    if verbose:
        print(f"M_int.shape: {M_int.shape}")
        print(f"X_int_scatter.shape: {X_int_scatter.shape}")
        print(f"C_int.shape: {C_int.shape}")
        print(f"C_int_scatter.shape: {C_int_scatter.shape}")
        print(f"tau_EBV.shape: {tau_EBV.shape}\n")

    # Define priors on shared population parameters

    alpha = npy.sample("alpha", dist.Normal(0, 5.0))
    beta = npy.sample("beta", dist.Normal(0, 5.0))
    R_B = npy.sample("R_B", dist.HalfNormal(5.0))
    R_B_scatter = npy.sample("R_B_scatter", dist.HalfNormal(5.0))
    unshifted_gamma_EBV = npy.sample("unshifted_gamma_EBV", dist.LogNormal(0.0, 5.0))
    gamma_EBV = npy.deterministic("gamma_EBV", unshifted_gamma_EBV + 1)

    if verbose:
        print(f"alpha.shape: {alpha.shape}")
        print(f"beta.shape: {beta.shape}")
        print(f"R_B.shape: {R_B.shape}")
        print(f"R_B_scatter.shape: {R_B_scatter.shape}")
        print(f"unshifted_gamma_EBV.shape: {unshifted_gamma_EBV.shape}")
        print(f"gamma_EBV.shape: {gamma_EBV.shape}")

    # Priors on Host Mass
    M_host_mean = 10.5 - host_mean
    M_host = npy.sample("M_host", dist.Normal(M_host_mean, 5.0))
    M_host_scatter = npy.sample("M_host_scatter", dist.HalfNormal(5.0))

    if verbose:
        print(f"M_host.shape: {M_host.shape}")
        print(f"M_host_scatter.shape: {M_host_scatter.shape}")

    # Priors on Host Mass - based SN Population Fraction
    scaling = npy.sample("scaling", dist.Normal(0.0, 5.0))
    offset = 0.0  # npy.sample("offset", dist.Normal(0.0, 5.0))
    f_SN_1_mid = npy.sample(
        "f_SN_1_mid", dist.Uniform(f_SN_1_min, 0.5)  # - f_SN_1_min / 2)
    )
    f_SN_1_max = npy.deterministic("f_SN_1_max", 2 * f_SN_1_mid)

    if verbose:
        print(f"scaling.shape: {scaling.shape}")
        # print(f"offset.shape: {offset.shape}")
        print(f"f_SN_1_mid.shape: {f_SN_1_mid.shape}\n")

    with npy.plate("sn", size=len(sn_redshifts)):
        # Latent Host Mass
        M_host_latent = npy.sample("M_host_latent", dist.Normal(M_host, M_host_scatter))

        if host_mass is not None:
            if verbose:
                print(
                    f"Standardizing Host Mass using mean: {host_mean} and std: {host_std}"
                )
            obs_host_mass = (host_mass - host_mean) / host_std
        else:
            obs_host_mass = None
        obs_host_mass_err = host_mass_err / host_std

        sampled_host_observables = npy.sample(
            "host_observables",
            dist.Normal(M_host_latent, obs_host_mass_err),
            obs=obs_host_mass,
        )

        if verbose:
            print(f"M_host_latent.shape: {M_host_latent.shape}")
            print(f"sampled_host_observables.shape: {sampled_host_observables.shape}\n")

        # Mass - based SN Population Fraction
        rescaled_M_host = (M_host_latent - M_host) / M_host_scatter
        linear_function = npy.deterministic(
            "linear_function", scaling * rescaled_M_host + offset
        )
        f_sn_1 = npy.deterministic(
            "f_sn_1", sigmoid(linear_function, f_mid=f_SN_1_mid, f_min=f_SN_1_min)
        )
        f_sn_2 = npy.deterministic("f_sn_2", 1 - f_sn_1)
        f_sn = jnp.stack([f_sn_1, f_sn_2], axis=-1)

        if verbose:
            print(f"linear_function.shape: {linear_function.shape}")
            print(f"f_sn_1.shape: {f_sn_1.shape}")
            print(f"f_sn_2.shape: {f_sn_2.shape}")
            print(f"f_sn.shape: {f_sn.shape}\n")

        # SN Population assignment
        population_assignment = npy.sample(
            "population_assignment",
            dist.Categorical(f_sn),
            infer={"enumerate": "parallel"},
        )
        M_int_latent = Vindex(M_int)[..., population_assignment]

        if verbose:
            print(f"population_assignment.shape: {population_assignment.shape}")
            print(f"M_int_latent.shape: {M_int_latent.shape}\n")

        X_int_latent_SN_pop_1 = npy.sample(
            "X_int_latent_SN_pop_1",
            dist.Normal(Vindex(X_int)[..., 0], Vindex(X_int_scatter)[..., 0]),
        )
        X_int_latent_SN_pop_2 = npy.sample(
            "X_int_latent_SN_pop_2",
            dist.Normal(Vindex(X_int)[..., 1], Vindex(X_int_scatter)[..., 1]),
        )
        X_int_latent_values = jnp.stack(
            [X_int_latent_SN_pop_1, X_int_latent_SN_pop_2], axis=-1
        )
        X_int_latent = npy.deterministic(
            "X_int_latent", Vindex(X_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"X_int_latent_SN_pop_1.shape: {X_int_latent_SN_pop_1.shape}")
            print(f"X_int_latent_SN_pop_2.shape: {X_int_latent_SN_pop_2.shape}")
            print(f"X_int_latent_values.shape: {X_int_latent_values.shape}")
            print(f"X_int_latent.shape: {X_int_latent.shape}\n")

        C_int_latent_SN_pop_1 = npy.sample(
            "C_int_latent_SN_pop_1",
            dist.Normal(Vindex(C_int)[..., 0], Vindex(C_int_scatter)[..., 0]),
        )
        C_int_latent_SN_pop_2 = npy.sample(
            "C_int_latent_SN_pop_2",
            dist.Normal(Vindex(C_int)[..., 1], Vindex(C_int_scatter)[..., 1]),
        )
        C_int_latent_values = jnp.stack(
            [C_int_latent_SN_pop_1, C_int_latent_SN_pop_2], axis=-1
        )
        C_int_latent = npy.deterministic(
            "C_int_latent", Vindex(C_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"C_int_latent_SN_pop_1.shape: {C_int_latent_SN_pop_1.shape}")
            print(f"C_int_latent_SN_pop_2.shape: {C_int_latent_SN_pop_2.shape}")
            print(f"C_int_latent_values.shape: {C_int_latent_values.shape}")
            print(f"C_int_latent.shape: {C_int_latent.shape}\n")

        EBV_latent_decentered = npy.sample(
            "EBV_latent_decentered", dist.Gamma(gamma_EBV)
        )
        EBV_latent_SN_pop_1 = npy.deterministic(
            "EBV_latent_SN_pop_1", EBV_latent_decentered * Vindex(tau_EBV)[..., 0]
        )
        EBV_latent_SN_pop_2 = npy.deterministic(
            "EBV_latent_SN_pop_2", EBV_latent_decentered * Vindex(tau_EBV)[..., 1]
        )
        EBV_latent_values = jnp.stack(
            [EBV_latent_SN_pop_1, EBV_latent_SN_pop_2], axis=-1
        )
        EBV_latent = npy.deterministic(
            "EBV_latent", Vindex(EBV_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"EBV_latent_decentered.shape: {EBV_latent_decentered.shape}")
            print(f"EBV_latent_SN_pop_1.shape: {EBV_latent_SN_pop_1.shape}")
            print(f"EBV_latent_SN_pop_2.shape: {EBV_latent_SN_pop_2.shape}")
            print(f"EBV_latent_values.shape: {EBV_latent_values.shape}")
            print(f"EBV_latent.shape: {EBV_latent.shape}\n")

        R_B_latent = npy.sample(
            "R_B_latent",
            dist.Normal(
                loc=R_B,
                scale=R_B_scatter,  # low=1,
            ),
        )

        if verbose:
            print(f"R_B_latent.shape: {R_B_latent.shape}\n")

        distance_modulus = distance_moduli(cosmo, sn_redshifts)

        if verbose:
            print(f"distance_modulus.shape: {distance_modulus.shape}\n")

        app_mag_1 = (
            Vindex(M_int)[..., 0]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_1
            + beta * C_int_latent_SN_pop_1
            + R_B_latent * EBV_latent_SN_pop_1
        )
        app_stretch_1 = X_int_latent_SN_pop_1
        app_color_1 = C_int_latent_SN_pop_1 + EBV_latent_SN_pop_1
        mean_1 = jnp.stack([app_mag_1, app_stretch_1, app_color_1], axis=-1)
        sn_dist_1 = dist.MultivariateNormal(mean_1, sn_covariances)

        if sn_observables is not None:
            sn_log_prob_1 = sn_dist_1.log_prob(sn_observables) + jnp.log(f_sn_1)

        if verbose:
            print(f"app_mag_1.shape: {app_mag_1.shape}")
            print(f"app_stretch_1.shape: {app_stretch_1.shape}")
            print(f"app_color_1.shape: {app_color_1.shape}")
            print(f"mean_1.shape: {mean_1.shape}")
            print(f"sn_dist_1.shape: {sn_dist_1.shape()}")
            print(f"sn_log_prob_1.shape: {sn_log_prob_1.shape}\n")

        app_mag_2 = (
            Vindex(M_int)[..., 1]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_2
            + beta * C_int_latent_SN_pop_2
            + R_B_latent * EBV_latent_SN_pop_2
        )
        app_stretch_2 = X_int_latent_SN_pop_2
        app_color_2 = C_int_latent_SN_pop_2 + EBV_latent_SN_pop_2
        mean_2 = jnp.stack([app_mag_2, app_stretch_2, app_color_2], axis=-1)
        sn_dist_2 = dist.MultivariateNormal(mean_2, sn_covariances)
        if sn_observables is not None:
            sn_log_prob_2 = sn_dist_2.log_prob(sn_observables) + jnp.log(1 - f_sn_1)

        if verbose:
            print(f"app_mag_2.shape: {app_mag_2.shape}")
            print(f"app_stretch_2.shape: {app_stretch_2.shape}")
            print(f"app_color_2.shape: {app_color_2.shape}")
            print(f"mean_2.shape: {mean_2.shape}")
            print(f"sn_dist_2.shape: {sn_dist_2.shape()}")
            print(f"sn_log_prob_2.shape: {sn_log_prob_2.shape}\n")

        apparent_magnitude_values = jnp.stack([app_mag_1, app_mag_2], axis=-1).squeeze()

        if verbose:
            print(f"apparent_magnitude_values.shape: {apparent_magnitude_values.shape}")

        if sn_observables is not None:
            npy.deterministic(
                "sn_log_membership_ratio", (sn_log_prob_1 - sn_log_prob_2) / jnp.log(10)
            )

        apparent_magnitude = npy.deterministic(
            "apparent_magnitude",
            Vindex(apparent_magnitude_values)[..., population_assignment],
        )
        apparent_stretch = npy.deterministic("apparent_stretch", X_int_latent)
        apparent_color = npy.deterministic("apparent_color", C_int_latent + EBV_latent)

        if verbose:
            print(f"apparent_magnitude.shape: {apparent_magnitude.shape}")
            print(f"apparent_stretch.shape: {apparent_stretch.shape}")
            print(f"apparent_color.shape: {apparent_color.shape}")

        # Sample observables
        mean = jnp.stack(
            [apparent_magnitude, apparent_stretch, apparent_color], axis=-1
        )

        if verbose:
            print(f"mean.shape: {mean.shape}\n")
            print(f"sn_observables.shape: {sn_observables.shape}")
            print(f"sn_covariances.shape: {sn_covariances.shape}\n")

        result_dist = dist.MultivariateNormal(mean, sn_covariances)
        npy.sample("sn_observables", result_dist, obs=sn_observables)


SN2PopMass_reparam_cfg = {
    "X_int_latent": LocScaleReparam(0.0),
    "C_int_latent": LocScaleReparam(0.0),
    # "R_B_latent": LocScaleReparam(0.0),
    # "M_host_latent": LocScaleReparam(0.0),
}


@config_enumerate
@reparam(config=SN2PopMass_reparam_cfg)
def SN2PopMass(
    sn_observables=None,
    sn_covariances=None,
    sn_redshifts=None,
    host_mass=None,
    host_mass_err=None,
    cosmology=None,
    constraint_factor=-500,
    verbose=False,
    host_mean=0.0,
    host_std=1.0,
    f_SN_1_min=0.15,
    lower_host_mass_bound=None,
    upper_host_mass_bound=None,
    lower_R_B_bound=None,
    upper_R_B_bound=None,
    *args,
    **kwargs,
):
    # sn_observables: [N, 3] array of observables
    # sn_covariance: [N, 3, 3] array of covariance matrices
    # sn_redshifts: [N] array of redshifts
    # host_mass: [N] array of host masses

    # Define Cosmology
    cosmo = cosmology

    # Define priors on independent population parameters

    if verbose:
        print("\n---------------- MODEL SHAPES ----------------")

    X_int = npy.sample(
        "X_int",
        dist.TransformedDistribution(
            dist.Normal(jnp.array([0.0, 0.0]), jnp.array([5.0, 5.0])),
            OrderedTransform(),
        ),
    )

    if verbose:
        print(f"X_int.shape: {X_int.shape}")

    with npy.plate("sn_populations", size=2):
        M_int = npy.sample(
            "M_int", dist.TruncatedNormal(-19.5, 1, high=-15.0, low=-25.0)
        )
        X_int_scatter = npy.sample("X_int_scatter", dist.HalfNormal(5.0))
        C_int = npy.sample("C_int", dist.Normal(0.0, 5.0))
        C_int_scatter = npy.sample("C_int_scatter", dist.HalfNormal(5.0))
        tau_EBV = npy.sample("tau_EBV", dist.HalfNormal(5.0))

    if verbose:
        print(f"M_int.shape: {M_int.shape}")
        print(f"X_int_scatter.shape: {X_int_scatter.shape}")
        print(f"C_int.shape: {C_int.shape}")
        print(f"C_int_scatter.shape: {C_int_scatter.shape}")
        print(f"tau_EBV.shape: {tau_EBV.shape}\n")

    # Define priors on shared population parameters

    alpha = npy.sample("alpha", dist.Normal(0, 5.0))
    beta = npy.sample("beta", dist.Normal(0, 5.0))
    # R_B = npy.sample("R_B", dist.HalfNormal(5.0))
    R_B = npy.sample(
        "R_B",
        dist.TruncatedNormal(
            4.1,
            3.0,
            low=lower_R_B_bound,
            high=upper_R_B_bound,
        ),
    )
    R_B_scatter = npy.sample("R_B_scatter", dist.HalfNormal(5.0))
    unshifted_gamma_EBV = npy.sample("unshifted_gamma_EBV", dist.LogNormal(0.0, 5.0))
    gamma_EBV = npy.deterministic("gamma_EBV", unshifted_gamma_EBV + 1)

    if verbose:
        print(f"alpha.shape: {alpha.shape}")
        print(f"beta.shape: {beta.shape}")
        print(f"R_B.shape: {R_B.shape}")
        print(f"R_B_scatter.shape: {R_B_scatter.shape}")
        print(f"unshifted_gamma_EBV.shape: {unshifted_gamma_EBV.shape}")
        print(f"gamma_EBV.shape: {gamma_EBV.shape}")

    # Priors on Host Mass
    M_host_mean = 10.5 - host_mean
    with npy.plate("host_populations", size=2):
        M_host = npy.sample("M_host", dist.Normal(M_host_mean, 5.0))
        M_host_scatter = npy.sample("M_host_scatter", dist.HalfNormal(5.0))

    f_host_1 = npy.sample("f_host_1", dist.Uniform(0.15, 0.85))
    f_host = jnp.array([f_host_1, 1 - f_host_1])

    M_host_mixture_mean = npy.deterministic(
        "M_host_mixture_mean",
        f_host_1 * Vindex(M_host)[..., 0] + (1 - f_host_1) * Vindex(M_host)[..., 1],
    )
    M_host_mixture_std = npy.deterministic(
        "M_host_mixture_std",
        jnp.sqrt(
            f_host_1 * Vindex(M_host_scatter)[..., 0] ** 2
            + (1 - f_host_1) * Vindex(M_host_scatter)[..., 1] ** 2
            + +f_host_1
            * (1 - f_host_1)
            * (Vindex(M_host)[..., 0] - Vindex(M_host)[..., 1]) ** 2
        ),
    )

    if verbose:
        print(f"M_host.shape: {M_host.shape}")
        print(f"M_host_scatter.shape: {M_host_scatter.shape}")
        print(f"f_host_1.shape: {f_host_1.shape}")
        print(f"f_host.shape: {f_host.shape}")
        print(f"M_host_mixture_mean.shape: {M_host_mixture_mean.shape}")
        print(f"M_host_mixture_std.shape: {M_host_mixture_std.shape}")

    # Priors on Host Mass - based SN Population Fraction
    scaling = npy.sample("scaling", dist.HalfNormal(5.0))
    offset = 0.0  # npy.sample("offset", dist.Normal(0.0, 5.0))
    f_SN_1_mid = npy.sample(
        "f_SN_1_mid", dist.Uniform(f_SN_1_min, 0.5)  # - f_SN_1_min / 2)
    )
    f_SN_1_max = npy.deterministic("f_SN_1_max", 2 * f_SN_1_mid)

    if verbose:
        print(f"scaling.shape: {scaling.shape}")
        # print(f"offset.shape: {offset.shape}")
        print(f"f_SN_1_mid.shape: {f_SN_1_mid.shape}\n")

    with npy.plate("sn", size=len(sn_redshifts)):
        # Host Population Assignment
        host_population_assignment = npy.sample(
            "host_population_assignment",
            dist.Categorical(f_host),
            infer={"enumerate": "parallel"},
        )
        M_host_latent_pop_1 = npy.sample(
            "M_host_latent_pop_1",
            dist.TruncatedNormal(
                Vindex(M_host)[..., 0],
                Vindex(M_host_scatter)[..., 0],
                low=lower_host_mass_bound,
                high=upper_host_mass_bound,
            ),
        )
        M_host_latent_pop_2 = npy.sample(
            "M_host_latent_pop_2",
            dist.TruncatedNormal(
                Vindex(M_host)[..., 1],
                Vindex(M_host_scatter)[..., 1],
                low=lower_host_mass_bound,
                high=upper_host_mass_bound,
            ),
        )
        M_host_latent_values = jnp.stack(
            [M_host_latent_pop_1, M_host_latent_pop_2], axis=-1
        )
        M_host_latent = npy.deterministic(
            "M_host_latent",
            Vindex(M_host_latent_values)[..., host_population_assignment],
        )

        if host_mass is not None:
            if verbose:
                print(
                    f"Standardizing Host Mass using mean: {host_mean} and std: {host_std}"
                )
            obs_host_mass = (host_mass - host_mean) / host_std
        else:
            obs_host_mass = None
        obs_host_mass_err = host_mass_err / host_std

        print(f"Lower Host Mass Bound: {lower_host_mass_bound}")
        print(f"Upper Host Mass Bound: {upper_host_mass_bound}")
        print(f"Lower R_B Bound: {lower_R_B_bound}")
        print(f"Upper R_B Bound: {upper_R_B_bound}")

        sampled_host_observables = npy.sample(
            "host_observables",
            dist.TruncatedNormal(
                M_host_latent,
                obs_host_mass_err,
                low=lower_host_mass_bound,
                high=upper_host_mass_bound,
            ),
            obs=obs_host_mass,
        )

        if verbose:
            print(
                f"host_population_assignment.shape: {host_population_assignment.shape}"
            )
            print(f"M_host_latent_pop_1.shape: {M_host_latent_pop_1.shape}")
            print(f"M_host_latent_pop_2.shape: {M_host_latent_pop_2.shape}")
            print(f"M_host_latent_values.shape: {M_host_latent_values.shape}")
            print(f"M_host_latent.shape: {M_host_latent.shape}")
            print(f"sampled_host_observables.shape: {sampled_host_observables.shape}\n")

        # Mass - based SN Population Fraction
        rescaled_M_host = (M_host_latent - M_host_mixture_mean) / M_host_mixture_std
        linear_function = npy.deterministic(
            "linear_function", scaling * rescaled_M_host + offset
        )
        f_sn_1 = npy.deterministic(
            "f_sn_1", sigmoid(linear_function, f_mid=f_SN_1_mid, f_min=f_SN_1_min)
        )
        f_sn_2 = npy.deterministic("f_sn_2", 1 - f_sn_1)
        f_sn = jnp.stack([f_sn_1, f_sn_2], axis=-1)

        if verbose:
            print(f"rescaled_M_host.shape: {rescaled_M_host.shape}")
            print(f"linear_function.shape: {linear_function.shape}")
            print(f"f_sn_1.shape: {f_sn_1.shape}")
            print(f"f_sn_2.shape: {f_sn_2.shape}")
            print(f"f_sn.shape: {f_sn.shape}\n")

        # SN Population assignment
        population_assignment = npy.sample(
            "population_assignment",
            dist.Categorical(f_sn),
            infer={"enumerate": "parallel"},
        )
        M_int_latent = Vindex(M_int)[..., population_assignment]

        if verbose:
            print(f"population_assignment.shape: {population_assignment.shape}")
            print(f"M_int_latent.shape: {M_int_latent.shape}\n")

        X_int_latent_SN_pop_1 = npy.sample(
            "X_int_latent_SN_pop_1",
            dist.Normal(Vindex(X_int)[..., 0], Vindex(X_int_scatter)[..., 0]),
        )
        X_int_latent_SN_pop_2 = npy.sample(
            "X_int_latent_SN_pop_2",
            dist.Normal(Vindex(X_int)[..., 1], Vindex(X_int_scatter)[..., 1]),
        )
        X_int_latent_values = jnp.stack(
            [X_int_latent_SN_pop_1, X_int_latent_SN_pop_2], axis=-1
        )
        X_int_latent = npy.deterministic(
            "X_int_latent", Vindex(X_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"X_int_latent_SN_pop_1.shape: {X_int_latent_SN_pop_1.shape}")
            print(f"X_int_latent_SN_pop_2.shape: {X_int_latent_SN_pop_2.shape}")
            print(f"X_int_latent_values.shape: {X_int_latent_values.shape}")
            print(f"X_int_latent.shape: {X_int_latent.shape}\n")

        C_int_latent_SN_pop_1 = npy.sample(
            "C_int_latent_SN_pop_1",
            dist.Normal(Vindex(C_int)[..., 0], Vindex(C_int_scatter)[..., 0]),
        )
        C_int_latent_SN_pop_2 = npy.sample(
            "C_int_latent_SN_pop_2",
            dist.Normal(Vindex(C_int)[..., 1], Vindex(C_int_scatter)[..., 1]),
        )
        C_int_latent_values = jnp.stack(
            [C_int_latent_SN_pop_1, C_int_latent_SN_pop_2], axis=-1
        )
        C_int_latent = npy.deterministic(
            "C_int_latent", Vindex(C_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"C_int_latent_SN_pop_1.shape: {C_int_latent_SN_pop_1.shape}")
            print(f"C_int_latent_SN_pop_2.shape: {C_int_latent_SN_pop_2.shape}")
            print(f"C_int_latent_values.shape: {C_int_latent_values.shape}")
            print(f"C_int_latent.shape: {C_int_latent.shape}\n")

        EBV_latent_decentered = npy.sample(
            "EBV_latent_decentered", dist.Gamma(gamma_EBV)
        )
        EBV_latent_SN_pop_1 = npy.deterministic(
            "EBV_latent_SN_pop_1", EBV_latent_decentered * Vindex(tau_EBV)[..., 0]
        )
        EBV_latent_SN_pop_2 = npy.deterministic(
            "EBV_latent_SN_pop_2", EBV_latent_decentered * Vindex(tau_EBV)[..., 1]
        )
        EBV_latent_values = jnp.stack(
            [EBV_latent_SN_pop_1, EBV_latent_SN_pop_2], axis=-1
        )
        EBV_latent = npy.deterministic(
            "EBV_latent", Vindex(EBV_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"EBV_latent_decentered.shape: {EBV_latent_decentered.shape}")
            print(f"EBV_latent_SN_pop_1.shape: {EBV_latent_SN_pop_1.shape}")
            print(f"EBV_latent_SN_pop_2.shape: {EBV_latent_SN_pop_2.shape}")
            print(f"EBV_latent_values.shape: {EBV_latent_values.shape}")
            print(f"EBV_latent.shape: {EBV_latent.shape}\n")

        R_B_latent = npy.sample(
            "R_B_latent",
            dist.TruncatedNormal(loc=R_B, scale=R_B_scatter, low=lower_R_B_bound),
        )

        if verbose:
            print(f"R_B_latent.shape: {R_B_latent.shape}\n")

        distance_modulus = distance_moduli(cosmo, sn_redshifts)

        if verbose:
            print(f"distance_modulus.shape: {distance_modulus.shape}\n")

        app_mag_1 = (
            Vindex(M_int)[..., 0]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_1
            + beta * C_int_latent_SN_pop_1
            + R_B_latent * EBV_latent_SN_pop_1
        )
        app_stretch_1 = X_int_latent_SN_pop_1
        app_color_1 = C_int_latent_SN_pop_1 + EBV_latent_SN_pop_1
        mean_1 = jnp.stack([app_mag_1, app_stretch_1, app_color_1], axis=-1)
        sn_dist_1 = dist.MultivariateNormal(mean_1, sn_covariances)

        if sn_observables is not None:
            sn_log_prob_1 = sn_dist_1.log_prob(sn_observables) + jnp.log(f_sn_1)

        if verbose:
            print(f"app_mag_1.shape: {app_mag_1.shape}")
            print(f"app_stretch_1.shape: {app_stretch_1.shape}")
            print(f"app_color_1.shape: {app_color_1.shape}")
            print(f"mean_1.shape: {mean_1.shape}")
            print(f"sn_dist_1.shape: {sn_dist_1.shape()}")
            print(f"sn_log_prob_1.shape: {sn_log_prob_1.shape}\n")

        app_mag_2 = (
            Vindex(M_int)[..., 1]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_2
            + beta * C_int_latent_SN_pop_2
            + R_B_latent * EBV_latent_SN_pop_2
        )
        app_stretch_2 = X_int_latent_SN_pop_2
        app_color_2 = C_int_latent_SN_pop_2 + EBV_latent_SN_pop_2
        mean_2 = jnp.stack([app_mag_2, app_stretch_2, app_color_2], axis=-1)
        sn_dist_2 = dist.MultivariateNormal(mean_2, sn_covariances)
        if sn_observables is not None:
            sn_log_prob_2 = sn_dist_2.log_prob(sn_observables) + jnp.log(1 - f_sn_1)

        if verbose:
            print(f"app_mag_2.shape: {app_mag_2.shape}")
            print(f"app_stretch_2.shape: {app_stretch_2.shape}")
            print(f"app_color_2.shape: {app_color_2.shape}")
            print(f"mean_2.shape: {mean_2.shape}")
            print(f"sn_dist_2.shape: {sn_dist_2.shape()}")
            print(f"sn_log_prob_2.shape: {sn_log_prob_2.shape}\n")

        apparent_magnitude_values = jnp.stack([app_mag_1, app_mag_2], axis=-1).squeeze()

        if verbose:
            print(f"apparent_magnitude_values.shape: {apparent_magnitude_values.shape}")

        if sn_observables is not None:
            npy.deterministic(
                "sn_log_membership_ratio", (sn_log_prob_1 - sn_log_prob_2) / jnp.log(10)
            )

        apparent_magnitude = npy.deterministic(
            "apparent_magnitude",
            Vindex(apparent_magnitude_values)[..., population_assignment],
        )
        apparent_stretch = npy.deterministic("apparent_stretch", X_int_latent)
        apparent_color = npy.deterministic("apparent_color", C_int_latent + EBV_latent)

        if verbose:
            print(f"apparent_magnitude.shape: {apparent_magnitude.shape}")
            print(f"apparent_stretch.shape: {apparent_stretch.shape}")
            print(f"apparent_color.shape: {apparent_color.shape}")

        # Sample observables
        mean = jnp.stack(
            [apparent_magnitude, apparent_stretch, apparent_color], axis=-1
        )

        if verbose:
            print(f"mean.shape: {mean.shape}\n")
            print(f"sn_observables.shape: {sn_observables.shape}")
            print(f"sn_covariances.shape: {sn_covariances.shape}\n")

        result_dist = dist.MultivariateNormal(mean, sn_covariances)
        npy.sample("sn_observables", result_dist, obs=sn_observables)


# -------------------- GAUSSIAN PROCESS MODEL DEFINITIONS --------------------


def spectral_density(w, alpha, length):
    c = alpha * jnp.sqrt(2 * jnp.pi) * length
    e = jnp.exp(-0.5 * (length**2) * (w**2))
    return c * e


def diag_spectral_density(alpha, length, L, M):
    sqrt_eigenvalues = jnp.arange(1, 1 + M) * jnp.pi / 2 / L
    return spectral_density(sqrt_eigenvalues, alpha, length)


def eigenfunctions(x, L, M):
    """
    The first `M` eigenfunctions of the laplacian operator in `[-L, L]`
    evaluated at `x`. These are used for the approximation of the
    squared exponential kernel.
    """

    m1 = (jnp.pi / (2 * L)) * jnp.tile(L + x[:, None], M)
    m2 = jnp.diag(jnp.linspace(1, M, num=M))
    num = jnp.sin(m1 @ m2)
    den = jnp.sqrt(L)
    return num / den


def approx_se_ncp(x, alpha, length, L, M, verbose=False):
    """
    Hilbert space approximation for the squared
    exponential kernel in the non-centered parametrisation.
    """
    phi = eigenfunctions(x, L, M)
    spd = jnp.sqrt(diag_spectral_density(alpha, length, L, M))
    with npy.plate("gp_basis", M):
        beta = npy.sample("gp_beta", dist.Normal(0, 1))

    f = npy.deterministic("gp", phi @ (spd * beta))

    if verbose:
        print(f"phi.shape: {phi.shape}")
        print(f"spd.shape: {spd.shape}")
        print(f"beta.shape: {beta.shape}")
        print(f"f.shape: {f.shape}\n")

    return f


def host_GP(x, bound_factor=6.4, minimum_components=20, verbose=False):
    S = jnp.max(jnp.abs(x))
    L = bound_factor * S
    normalized_length_scale = npy.sample(
        "normalized_length_scale", dist.Uniform(0.01, 2.0)
    )
    length_scale = npy.deterministic("gp_length", normalized_length_scale * S)
    sigma = npy.sample("gp_sigma", dist.HalfNormal(0.5))
    alpha = sigma**2

    M = 20

    # M = npy.deterministic(
    #     "n_components",
    #     jnp.maximum(
    #         minimum_components,
    #         (1.75 * bound_factor / normalized_length_scale + 5),
    #     ).astype(int),
    # )

    if verbose:
        print(f"normalized_length_scale.shape: {normalized_length_scale.shape}")
        print(f"length_scale.shape: {length_scale.shape}")
        print(f"sigma.shape: {sigma.shape}")

    f = approx_se_ncp(x=x, alpha=alpha, length=length_scale, L=L, M=M, verbose=verbose)

    return f


SNMassGP_reparam_cfg = {
    "X_int_latent": LocScaleReparam(0.0),
    "C_int_latent": LocScaleReparam(0.0),
    # "R_B_latent": LocScaleReparam(0.0),
    # "M_host_latent": LocScaleReparam(0.0),
}


@config_enumerate
@reparam(config=SNMassGP_reparam_cfg)
def SNMassGP(
    sn_observables=None,
    sn_covariances=None,
    sn_redshifts=None,
    host_mass=None,
    host_mass_err=None,
    cosmology=None,
    constraint_factor=-500,
    verbose=False,
    host_mean=0.0,
    host_std=1.0,
    f_SN_1_min=0.15,
    lower_host_mass_bound=None,
    upper_host_mass_bound=None,
    lower_R_B_bound=None,
    upper_R_B_bound=None,
    *args,
    **kwargs,
):
    # sn_observables: [N, 3] array of observables
    # sn_covariance: [N, 3, 3] array of covariance matrices
    # sn_redshifts: [N] array of redshifts
    # host_mass: [N] array of host masses

    # Define Cosmology
    cosmo = cosmology

    # Define priors on independent population parameters

    if verbose:
        print("\n---------------- MODEL SHAPES ----------------")

    X_int = npy.sample(
        "X_int",
        dist.TransformedDistribution(
            dist.Normal(jnp.array([0.0, 0.0]), jnp.array([5.0, 5.0])),
            OrderedTransform(),
        ),
    )

    if verbose:
        print(f"X_int.shape: {X_int.shape}")

    with npy.plate("sn_populations", size=2):
        M_int = npy.sample(
            "M_int", dist.TruncatedNormal(-19.5, 1, high=-15.0, low=-25.0)
        )
        X_int_scatter = npy.sample("X_int_scatter", dist.HalfNormal(5.0))
        C_int = npy.sample("C_int", dist.Normal(0.0, 5.0))
        C_int_scatter = npy.sample("C_int_scatter", dist.HalfNormal(5.0))
        tau_EBV = npy.sample("tau_EBV", dist.HalfNormal(5.0))

    if verbose:
        print(f"M_int.shape: {M_int.shape}")
        print(f"X_int_scatter.shape: {X_int_scatter.shape}")
        print(f"C_int.shape: {C_int.shape}")
        print(f"C_int_scatter.shape: {C_int_scatter.shape}")
        print(f"tau_EBV.shape: {tau_EBV.shape}\n")

    # Define priors on shared population parameters

    alpha = npy.sample("alpha", dist.Normal(0, 5.0))
    beta = npy.sample("beta", dist.Normal(0, 5.0))
    R_B = npy.sample(
        "R_B",
        dist.TruncatedNormal(
            4.1,
            3.0,
            low=lower_R_B_bound,
            high=upper_R_B_bound,
        ),
    )
    R_B_scatter = npy.sample("R_B_scatter", dist.HalfNormal(5.0))
    unshifted_gamma_EBV = npy.sample("unshifted_gamma_EBV", dist.LogNormal(0.0, 5.0))
    gamma_EBV = npy.deterministic("gamma_EBV", unshifted_gamma_EBV + 1)

    if verbose:
        print(f"alpha.shape: {alpha.shape}")
        print(f"beta.shape: {beta.shape}")
        print(f"R_B.shape: {R_B.shape}")
        print(f"R_B_scatter.shape: {R_B_scatter.shape}")
        print(f"unshifted_gamma_EBV.shape: {unshifted_gamma_EBV.shape}")
        print(f"gamma_EBV.shape: {gamma_EBV.shape}")

    # Priors on Host Mass
    M_host_mean = 10.5 - host_mean
    M_host = npy.sample("M_host", dist.Normal(M_host_mean, 5.0))
    M_host_scatter = npy.sample("M_host_scatter", dist.HalfNormal(5.0))

    if verbose:
        print(f"M_host.shape: {M_host.shape}")
        print(f"M_host_scatter.shape: {M_host_scatter.shape}")

    # Priors on Host Mass - based SN Population Fraction
    f_SN_1_mid = npy.sample(
        "f_SN_1_mid", dist.Uniform(f_SN_1_min, 0.425)  # - f_SN_1_min / 2)
    )
    f_SN_1_max = npy.deterministic("f_SN_1_max", 2 * f_SN_1_mid)

    if verbose:
        print(f"f_SN_1_mid.shape: {f_SN_1_mid.shape}\n")

    with npy.plate("masses", size=len(sn_redshifts)):
        # Latent Host Mass
        M_host_latent = npy.sample(
            "M_host_latent",
            dist.TruncatedNormal(
                M_host,
                M_host_scatter,
                low=lower_host_mass_bound,
                high=upper_host_mass_bound,
            ),
        )

        if host_mass is not None:
            if verbose:
                print(
                    f"Standardizing Host Mass using mean: {host_mean} and std: {host_std}"
                )
            obs_host_mass = (host_mass - host_mean) / host_std
        else:
            obs_host_mass = None
        obs_host_mass_err = host_mass_err / host_std

        sampled_host_observables = npy.sample(
            "host_observables",
            dist.Normal(M_host_latent, obs_host_mass_err),
            obs=obs_host_mass,
        )

        if verbose:
            print(f"M_host_latent.shape: {M_host_latent.shape}")
            print(f"sampled_host_observables.shape: {sampled_host_observables.shape}\n")

    # Mass - based SN Population Fraction
    eval_mass = npy.deterministic("eval_mass", jnp.linspace(6, 12, 100))
    rescaled_eval_mass = (eval_mass - M_host_mean) / M_host_scatter
    rescaled_M_host = (M_host_latent - M_host) / M_host_scatter
    host_masses = jnp.concatenate([rescaled_M_host, rescaled_eval_mass], axis=0)
    # gp = host_GP(rescaled_M_host, verbose=verbose)
    gp = host_GP(host_masses, verbose=verbose)
    gp_obs = gp[: len(rescaled_M_host)]

    if verbose:
        print(f"gp_obs.shape: {gp_obs.shape}")

    f_sn_1 = npy.deterministic(
        "f_sn_1", sigmoid(gp_obs, f_mid=f_SN_1_mid, f_min=f_SN_1_min)
    )
    f_sn_2 = npy.deterministic("f_sn_2", 1 - f_sn_1)
    f_sn = jnp.stack([f_sn_1, f_sn_2], axis=-1)

    if verbose:
        print(f"gp.shape: {gp.shape}")
        print(f"f_sn_1.shape: {f_sn_1.shape}")
        print(f"f_sn_2.shape: {f_sn_2.shape}")
        print(f"f_sn.shape: {f_sn.shape}\n")

    with npy.plate("sn", size=len(sn_redshifts)):
        # SN Population assignment
        population_assignment = npy.sample(
            "population_assignment",
            dist.Categorical(f_sn),
            infer={"enumerate": "parallel"},
        )
        M_int_latent = Vindex(M_int)[..., population_assignment]

        if verbose:
            print(f"population_assignment.shape: {population_assignment.shape}")
            print(f"M_int_latent.shape: {M_int_latent.shape}\n")

        X_int_latent_SN_pop_1 = npy.sample(
            "X_int_latent_SN_pop_1",
            dist.Normal(Vindex(X_int)[..., 0], Vindex(X_int_scatter)[..., 0]),
        )
        X_int_latent_SN_pop_2 = npy.sample(
            "X_int_latent_SN_pop_2",
            dist.Normal(Vindex(X_int)[..., 1], Vindex(X_int_scatter)[..., 1]),
        )
        X_int_latent_values = jnp.stack(
            [X_int_latent_SN_pop_1, X_int_latent_SN_pop_2], axis=-1
        )
        X_int_latent = npy.deterministic(
            "X_int_latent", Vindex(X_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"X_int_latent_SN_pop_1.shape: {X_int_latent_SN_pop_1.shape}")
            print(f"X_int_latent_SN_pop_2.shape: {X_int_latent_SN_pop_2.shape}")
            print(f"X_int_latent_values.shape: {X_int_latent_values.shape}")
            print(f"X_int_latent.shape: {X_int_latent.shape}\n")

        C_int_latent_SN_pop_1 = npy.sample(
            "C_int_latent_SN_pop_1",
            dist.Normal(Vindex(C_int)[..., 0], Vindex(C_int_scatter)[..., 0]),
        )
        C_int_latent_SN_pop_2 = npy.sample(
            "C_int_latent_SN_pop_2",
            dist.Normal(Vindex(C_int)[..., 1], Vindex(C_int_scatter)[..., 1]),
        )
        C_int_latent_values = jnp.stack(
            [C_int_latent_SN_pop_1, C_int_latent_SN_pop_2], axis=-1
        )
        C_int_latent = npy.deterministic(
            "C_int_latent", Vindex(C_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"C_int_latent_SN_pop_1.shape: {C_int_latent_SN_pop_1.shape}")
            print(f"C_int_latent_SN_pop_2.shape: {C_int_latent_SN_pop_2.shape}")
            print(f"C_int_latent_values.shape: {C_int_latent_values.shape}")
            print(f"C_int_latent.shape: {C_int_latent.shape}\n")

        EBV_latent_decentered = npy.sample(
            "EBV_latent_decentered", dist.Gamma(gamma_EBV)
        )
        EBV_latent_SN_pop_1 = npy.deterministic(
            "EBV_latent_SN_pop_1", EBV_latent_decentered * Vindex(tau_EBV)[..., 0]
        )
        EBV_latent_SN_pop_2 = npy.deterministic(
            "EBV_latent_SN_pop_2", EBV_latent_decentered * Vindex(tau_EBV)[..., 1]
        )
        EBV_latent_values = jnp.stack(
            [EBV_latent_SN_pop_1, EBV_latent_SN_pop_2], axis=-1
        )
        EBV_latent = npy.deterministic(
            "EBV_latent", Vindex(EBV_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"EBV_latent_decentered.shape: {EBV_latent_decentered.shape}")
            print(f"EBV_latent_SN_pop_1.shape: {EBV_latent_SN_pop_1.shape}")
            print(f"EBV_latent_SN_pop_2.shape: {EBV_latent_SN_pop_2.shape}")
            print(f"EBV_latent_values.shape: {EBV_latent_values.shape}")
            print(f"EBV_latent.shape: {EBV_latent.shape}\n")

        R_B_latent = npy.sample(
            "R_B_latent",
            dist.TruncatedNormal(loc=R_B, scale=R_B_scatter, low=lower_R_B_bound),
        )

        if verbose:
            print(f"R_B_latent.shape: {R_B_latent.shape}\n")

        distance_modulus = distance_moduli(cosmo, sn_redshifts)

        if verbose:
            print(f"distance_modulus.shape: {distance_modulus.shape}\n")

        app_mag_1 = (
            Vindex(M_int)[..., 0]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_1
            + beta * C_int_latent_SN_pop_1
            + R_B_latent * EBV_latent_SN_pop_1
        )
        app_stretch_1 = X_int_latent_SN_pop_1
        app_color_1 = C_int_latent_SN_pop_1 + EBV_latent_SN_pop_1
        mean_1 = jnp.stack([app_mag_1, app_stretch_1, app_color_1], axis=-1)
        sn_dist_1 = dist.MultivariateNormal(mean_1, sn_covariances)

        if sn_observables is not None:
            sn_log_prob_1 = sn_dist_1.log_prob(sn_observables) + jnp.log(f_sn_1)

        if verbose:
            print(f"app_mag_1.shape: {app_mag_1.shape}")
            print(f"app_stretch_1.shape: {app_stretch_1.shape}")
            print(f"app_color_1.shape: {app_color_1.shape}")
            print(f"mean_1.shape: {mean_1.shape}")
            print(f"sn_dist_1.shape: {sn_dist_1.shape()}")
            print(f"sn_log_prob_1.shape: {sn_log_prob_1.shape}\n")

        app_mag_2 = (
            Vindex(M_int)[..., 1]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_2
            + beta * C_int_latent_SN_pop_2
            + R_B_latent * EBV_latent_SN_pop_2
        )
        app_stretch_2 = X_int_latent_SN_pop_2
        app_color_2 = C_int_latent_SN_pop_2 + EBV_latent_SN_pop_2
        mean_2 = jnp.stack([app_mag_2, app_stretch_2, app_color_2], axis=-1)
        sn_dist_2 = dist.MultivariateNormal(mean_2, sn_covariances)
        if sn_observables is not None:
            sn_log_prob_2 = sn_dist_2.log_prob(sn_observables) + jnp.log(1 - f_sn_1)

        if verbose:
            print(f"app_mag_2.shape: {app_mag_2.shape}")
            print(f"app_stretch_2.shape: {app_stretch_2.shape}")
            print(f"app_color_2.shape: {app_color_2.shape}")
            print(f"mean_2.shape: {mean_2.shape}")
            print(f"sn_dist_2.shape: {sn_dist_2.shape()}")
            print(f"sn_log_prob_2.shape: {sn_log_prob_2.shape}\n")

        apparent_magnitude_values = jnp.stack([app_mag_1, app_mag_2], axis=-1).squeeze()

        if verbose:
            print(f"apparent_magnitude_values.shape: {apparent_magnitude_values.shape}")

        if sn_observables is not None:
            npy.deterministic(
                "sn_log_membership_ratio", (sn_log_prob_1 - sn_log_prob_2) / jnp.log(10)
            )

        apparent_magnitude = npy.deterministic(
            "apparent_magnitude",
            Vindex(apparent_magnitude_values)[..., population_assignment],
        )
        apparent_stretch = npy.deterministic("apparent_stretch", X_int_latent)
        apparent_color = npy.deterministic("apparent_color", C_int_latent + EBV_latent)

        if verbose:
            print(f"apparent_magnitude.shape: {apparent_magnitude.shape}")
            print(f"apparent_stretch.shape: {apparent_stretch.shape}")
            print(f"apparent_color.shape: {apparent_color.shape}")

        # Sample observables
        mean = jnp.stack(
            [apparent_magnitude, apparent_stretch, apparent_color], axis=-1
        )

        if verbose:
            print(f"mean.shape: {mean.shape}\n")
            print(f"sn_observables.shape: {sn_observables.shape}")
            print(f"sn_covariances.shape: {sn_covariances.shape}\n")

        result_dist = dist.MultivariateNormal(mean, sn_covariances)
        npy.sample("sn_observables", result_dist, obs=sn_observables)


SN2PopMassGP_reparam_cfg = {
    "X_int_latent": LocScaleReparam(0.0),
    "C_int_latent": LocScaleReparam(0.0),
    # "R_B_latent": LocScaleReparam(0.0),
    # "M_host_latent": LocScaleReparam(0.0),
}


@config_enumerate
@reparam(config=SN2PopMassGP_reparam_cfg)
def SN2PopMassGP(
    sn_observables=None,
    sn_covariances=None,
    sn_redshifts=None,
    host_mass=None,
    host_mass_err=None,
    cosmology=None,
    constraint_factor=-500,
    verbose=False,
    host_mean=0.0,
    host_std=1.0,
    f_SN_1_min=0.15,
    lower_host_mass_bound=None,
    upper_host_mass_bound=None,
    lower_R_B_bound=None,
    upper_R_B_bound=None,
    *args,
    **kwargs,
):
    # sn_observables: [N, 3] array of observables
    # sn_covariance: [N, 3, 3] array of covariance matrices
    # sn_redshifts: [N] array of redshifts
    # host_mass: [N] array of host masses

    # Define Cosmology
    cosmo = cosmology

    # Define priors on independent population parameters

    if verbose:
        print("\n---------------- MODEL SHAPES ----------------")

    X_int = npy.sample(
        "X_int",
        dist.TransformedDistribution(
            dist.Normal(jnp.array([0.0, 0.0]), jnp.array([5.0, 5.0])),
            OrderedTransform(),
        ),
    )

    if verbose:
        print(f"X_int.shape: {X_int.shape}")

    with npy.plate("sn_populations", size=2):
        M_int = npy.sample(
            "M_int", dist.TruncatedNormal(-19.5, 1, high=-15.0, low=-25.0)
        )
        X_int_scatter = npy.sample("X_int_scatter", dist.HalfNormal(5.0))
        C_int = npy.sample("C_int", dist.Normal(0.0, 5.0))
        C_int_scatter = npy.sample("C_int_scatter", dist.HalfNormal(5.0))
        tau_EBV = npy.sample("tau_EBV", dist.HalfNormal(5.0))

    if verbose:
        print(f"M_int.shape: {M_int.shape}")
        print(f"X_int_scatter.shape: {X_int_scatter.shape}")
        print(f"C_int.shape: {C_int.shape}")
        print(f"C_int_scatter.shape: {C_int_scatter.shape}")
        print(f"tau_EBV.shape: {tau_EBV.shape}\n")

    # Define priors on shared population parameters

    alpha = npy.sample("alpha", dist.Normal(0, 5.0))
    beta = npy.sample("beta", dist.Normal(0, 5.0))
    # R_B = npy.sample("R_B", dist.HalfNormal(5.0))
    R_B = npy.sample(
        "R_B",
        dist.TruncatedNormal(
            4.1,
            3.0,
            low=lower_R_B_bound,
            high=upper_R_B_bound,
        ),
    )
    R_B_scatter = npy.sample("R_B_scatter", dist.HalfNormal(5.0))
    unshifted_gamma_EBV = npy.sample("unshifted_gamma_EBV", dist.LogNormal(0.0, 5.0))
    gamma_EBV = npy.deterministic("gamma_EBV", unshifted_gamma_EBV + 1)

    if verbose:
        print(f"alpha.shape: {alpha.shape}")
        print(f"beta.shape: {beta.shape}")
        print(f"R_B.shape: {R_B.shape}")
        print(f"R_B_scatter.shape: {R_B_scatter.shape}")
        print(f"unshifted_gamma_EBV.shape: {unshifted_gamma_EBV.shape}")
        print(f"gamma_EBV.shape: {gamma_EBV.shape}")

    # Priors on Host Mass
    M_host_mean = 10.5 - host_mean
    with npy.plate("host_populations", size=2):
        M_host = npy.sample("M_host", dist.Normal(M_host_mean, 5.0))
        M_host_scatter = npy.sample("M_host_scatter", dist.HalfNormal(5.0))

    f_host_1 = npy.sample("f_host_1", dist.Uniform(0.15, 0.85))
    f_host = jnp.array([f_host_1, 1 - f_host_1])

    M_host_mixture_mean = npy.deterministic(
        "M_host_mixture_mean",
        f_host_1 * Vindex(M_host)[..., 0] + (1 - f_host_1) * Vindex(M_host)[..., 1],
    )
    M_host_mixture_std = npy.deterministic(
        "M_host_mixture_std",
        jnp.sqrt(
            f_host_1 * Vindex(M_host_scatter)[..., 0] ** 2
            + (1 - f_host_1) * Vindex(M_host_scatter)[..., 1] ** 2
            + +f_host_1
            * (1 - f_host_1)
            * (Vindex(M_host)[..., 0] - Vindex(M_host)[..., 1]) ** 2
        ),
    )

    if verbose:
        print(f"M_host.shape: {M_host.shape}")
        print(f"M_host_scatter.shape: {M_host_scatter.shape}")
        print(f"f_host_1.shape: {f_host_1.shape}")
        print(f"f_host.shape: {f_host.shape}")
        print(f"M_host_mixture_mean.shape: {M_host_mixture_mean.shape}")
        print(f"M_host_mixture_std.shape: {M_host_mixture_std.shape}")

    # Priors on Host Mass - based SN Population Fraction
    f_SN_1_mid = npy.sample(
        "f_SN_1_mid", dist.Uniform(f_SN_1_min, 0.5)  # - f_SN_1_min / 2)
    )
    f_SN_1_max = npy.deterministic("f_SN_1_max", 2 * f_SN_1_mid)

    if verbose:
        print(f"f_SN_1_mid.shape: {f_SN_1_mid.shape}\n")

    with npy.plate("masses", size=len(sn_redshifts)):
        # Host Population Assignment
        host_population_assignment = npy.sample(
            "host_population_assignment",
            dist.Categorical(f_host),
            infer={"enumerate": "parallel"},
        )
        M_host_latent_pop_1 = npy.sample(
            "M_host_latent_pop_1",
            dist.TruncatedNormal(
                Vindex(M_host)[..., 0],
                Vindex(M_host_scatter)[..., 0],
                low=lower_host_mass_bound,
                high=upper_host_mass_bound,
            ),
        )
        M_host_latent_pop_2 = npy.sample(
            "M_host_latent_pop_2",
            dist.TruncatedNormal(
                Vindex(M_host)[..., 1],
                Vindex(M_host_scatter)[..., 1],
                low=lower_host_mass_bound,
                high=upper_host_mass_bound,
            ),
        )
        M_host_latent_values = jnp.stack(
            [M_host_latent_pop_1, M_host_latent_pop_2], axis=-1
        )
        M_host_latent = npy.deterministic(
            "M_host_latent",
            Vindex(M_host_latent_values)[..., host_population_assignment],
        )

        if host_mass is not None:
            if verbose:
                print(
                    f"Standardizing Host Mass using mean: {host_mean} and std: {host_std}"
                )
            obs_host_mass = (host_mass - host_mean) / host_std
        else:
            obs_host_mass = None
        obs_host_mass_err = host_mass_err / host_std

        print(f"Lower Host Mass Bound: {lower_host_mass_bound}")
        print(f"Upper Host Mass Bound: {upper_host_mass_bound}")
        print(f"Lower R_B Bound: {lower_R_B_bound}")
        print(f"Upper R_B Bound: {upper_R_B_bound}")

        sampled_host_observables = npy.sample(
            "host_observables",
            dist.TruncatedNormal(
                M_host_latent,
                obs_host_mass_err,
                low=lower_host_mass_bound,
                high=upper_host_mass_bound,
            ),
            obs=obs_host_mass,
        )

        if verbose:
            print(
                f"host_population_assignment.shape: {host_population_assignment.shape}"
            )
            print(f"M_host_latent_pop_1.shape: {M_host_latent_pop_1.shape}")
            print(f"M_host_latent_pop_2.shape: {M_host_latent_pop_2.shape}")
            print(f"M_host_latent_values.shape: {M_host_latent_values.shape}")
            print(f"M_host_latent.shape: {M_host_latent.shape}")
            print(f"sampled_host_observables.shape: {sampled_host_observables.shape}\n")

    # Mass - based SN Population Fraction
    # eval_mass = npy.deterministic("eval_mass", jnp.linspace(6, 12, 100))
    # rescaled_eval_mass = (eval_mass - M_host_mixture_mean) / M_host_mixture_std
    rescaled_M_host = (M_host_latent - M_host_mixture_mean) / M_host_mixture_std
    host_masses = rescaled_M_host  # jnp.concatenate([rescaled_M_host, rescaled_eval_mass], axis=0)
    # gp = host_GP(rescaled_M_host, verbose=verbose)
    gp = host_GP(host_masses, verbose=verbose)
    gp_obs = gp  # [: len(rescaled_M_host)]

    if verbose:
        print(f"gp_obs.shape: {gp_obs.shape}")

    f_sn_1 = npy.deterministic(
        "f_sn_1", sigmoid(gp_obs, f_mid=f_SN_1_mid, f_min=f_SN_1_min)
    )
    f_sn_2 = npy.deterministic("f_sn_2", 1 - f_sn_1)
    f_sn = jnp.stack([f_sn_1, f_sn_2], axis=-1)

    if verbose:
        print(f"gp.shape: {gp.shape}")
        print(f"f_sn_1.shape: {f_sn_1.shape}")
        print(f"f_sn_2.shape: {f_sn_2.shape}")
        print(f"f_sn.shape: {f_sn.shape}\n")

    with npy.plate("sn", size=len(sn_redshifts)):
        # SN Population assignment
        population_assignment = npy.sample(
            "population_assignment",
            dist.Categorical(f_sn),
            infer={"enumerate": "parallel"},
        )
        M_int_latent = Vindex(M_int)[..., population_assignment]

        if verbose:
            print(f"population_assignment.shape: {population_assignment.shape}")
            print(f"M_int_latent.shape: {M_int_latent.shape}\n")

        X_int_latent_SN_pop_1 = npy.sample(
            "X_int_latent_SN_pop_1",
            dist.Normal(Vindex(X_int)[..., 0], Vindex(X_int_scatter)[..., 0]),
        )
        X_int_latent_SN_pop_2 = npy.sample(
            "X_int_latent_SN_pop_2",
            dist.Normal(Vindex(X_int)[..., 1], Vindex(X_int_scatter)[..., 1]),
        )
        X_int_latent_values = jnp.stack(
            [X_int_latent_SN_pop_1, X_int_latent_SN_pop_2], axis=-1
        )
        X_int_latent = npy.deterministic(
            "X_int_latent", Vindex(X_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"X_int_latent_SN_pop_1.shape: {X_int_latent_SN_pop_1.shape}")
            print(f"X_int_latent_SN_pop_2.shape: {X_int_latent_SN_pop_2.shape}")
            print(f"X_int_latent_values.shape: {X_int_latent_values.shape}")
            print(f"X_int_latent.shape: {X_int_latent.shape}\n")

        C_int_latent_SN_pop_1 = npy.sample(
            "C_int_latent_SN_pop_1",
            dist.Normal(Vindex(C_int)[..., 0], Vindex(C_int_scatter)[..., 0]),
        )
        C_int_latent_SN_pop_2 = npy.sample(
            "C_int_latent_SN_pop_2",
            dist.Normal(Vindex(C_int)[..., 1], Vindex(C_int_scatter)[..., 1]),
        )
        C_int_latent_values = jnp.stack(
            [C_int_latent_SN_pop_1, C_int_latent_SN_pop_2], axis=-1
        )
        C_int_latent = npy.deterministic(
            "C_int_latent", Vindex(C_int_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"C_int_latent_SN_pop_1.shape: {C_int_latent_SN_pop_1.shape}")
            print(f"C_int_latent_SN_pop_2.shape: {C_int_latent_SN_pop_2.shape}")
            print(f"C_int_latent_values.shape: {C_int_latent_values.shape}")
            print(f"C_int_latent.shape: {C_int_latent.shape}\n")

        EBV_latent_decentered = npy.sample(
            "EBV_latent_decentered", dist.Gamma(gamma_EBV)
        )
        EBV_latent_SN_pop_1 = npy.deterministic(
            "EBV_latent_SN_pop_1", EBV_latent_decentered * Vindex(tau_EBV)[..., 0]
        )
        EBV_latent_SN_pop_2 = npy.deterministic(
            "EBV_latent_SN_pop_2", EBV_latent_decentered * Vindex(tau_EBV)[..., 1]
        )
        EBV_latent_values = jnp.stack(
            [EBV_latent_SN_pop_1, EBV_latent_SN_pop_2], axis=-1
        )
        EBV_latent = npy.deterministic(
            "EBV_latent", Vindex(EBV_latent_values)[..., population_assignment]
        )

        if verbose:
            print(f"EBV_latent_decentered.shape: {EBV_latent_decentered.shape}")
            print(f"EBV_latent_SN_pop_1.shape: {EBV_latent_SN_pop_1.shape}")
            print(f"EBV_latent_SN_pop_2.shape: {EBV_latent_SN_pop_2.shape}")
            print(f"EBV_latent_values.shape: {EBV_latent_values.shape}")
            print(f"EBV_latent.shape: {EBV_latent.shape}\n")

        R_B_latent = npy.sample(
            "R_B_latent",
            dist.TruncatedNormal(loc=R_B, scale=R_B_scatter, low=lower_R_B_bound),
        )

        if verbose:
            print(f"R_B_latent.shape: {R_B_latent.shape}\n")

        distance_modulus = distance_moduli(cosmo, sn_redshifts)

        if verbose:
            print(f"distance_modulus.shape: {distance_modulus.shape}\n")

        app_mag_1 = (
            Vindex(M_int)[..., 0]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_1
            + beta * C_int_latent_SN_pop_1
            + R_B_latent * EBV_latent_SN_pop_1
        )
        app_stretch_1 = X_int_latent_SN_pop_1
        app_color_1 = C_int_latent_SN_pop_1 + EBV_latent_SN_pop_1
        mean_1 = jnp.stack([app_mag_1, app_stretch_1, app_color_1], axis=-1)
        sn_dist_1 = dist.MultivariateNormal(mean_1, sn_covariances)

        if sn_observables is not None:
            sn_log_prob_1 = sn_dist_1.log_prob(sn_observables) + jnp.log(f_sn_1)

        if verbose:
            print(f"app_mag_1.shape: {app_mag_1.shape}")
            print(f"app_stretch_1.shape: {app_stretch_1.shape}")
            print(f"app_color_1.shape: {app_color_1.shape}")
            print(f"mean_1.shape: {mean_1.shape}")
            print(f"sn_dist_1.shape: {sn_dist_1.shape()}")
            print(f"sn_log_prob_1.shape: {sn_log_prob_1.shape}\n")

        app_mag_2 = (
            Vindex(M_int)[..., 1]
            + distance_modulus
            + alpha * X_int_latent_SN_pop_2
            + beta * C_int_latent_SN_pop_2
            + R_B_latent * EBV_latent_SN_pop_2
        )
        app_stretch_2 = X_int_latent_SN_pop_2
        app_color_2 = C_int_latent_SN_pop_2 + EBV_latent_SN_pop_2
        mean_2 = jnp.stack([app_mag_2, app_stretch_2, app_color_2], axis=-1)
        sn_dist_2 = dist.MultivariateNormal(mean_2, sn_covariances)
        if sn_observables is not None:
            sn_log_prob_2 = sn_dist_2.log_prob(sn_observables) + jnp.log(1 - f_sn_1)

        if verbose:
            print(f"app_mag_2.shape: {app_mag_2.shape}")
            print(f"app_stretch_2.shape: {app_stretch_2.shape}")
            print(f"app_color_2.shape: {app_color_2.shape}")
            print(f"mean_2.shape: {mean_2.shape}")
            print(f"sn_dist_2.shape: {sn_dist_2.shape()}")
            print(f"sn_log_prob_2.shape: {sn_log_prob_2.shape}\n")

        apparent_magnitude_values = jnp.stack([app_mag_1, app_mag_2], axis=-1).squeeze()

        if verbose:
            print(f"apparent_magnitude_values.shape: {apparent_magnitude_values.shape}")

        if sn_observables is not None:
            npy.deterministic(
                "sn_log_membership_ratio", (sn_log_prob_1 - sn_log_prob_2) / jnp.log(10)
            )

        apparent_magnitude = npy.deterministic(
            "apparent_magnitude",
            Vindex(apparent_magnitude_values)[..., population_assignment],
        )
        apparent_stretch = npy.deterministic("apparent_stretch", X_int_latent)
        apparent_color = npy.deterministic("apparent_color", C_int_latent + EBV_latent)

        if verbose:
            print(f"apparent_magnitude.shape: {apparent_magnitude.shape}")
            print(f"apparent_stretch.shape: {apparent_stretch.shape}")
            print(f"apparent_color.shape: {apparent_color.shape}")

        # Sample observables
        mean = jnp.stack(
            [apparent_magnitude, apparent_stretch, apparent_color], axis=-1
        )

        if verbose:
            print(f"mean.shape: {mean.shape}\n")
            print(f"sn_observables.shape: {sn_observables.shape}")
            print(f"sn_covariances.shape: {sn_covariances.shape}\n")

        result_dist = dist.MultivariateNormal(mean, sn_covariances)
        npy.sample("sn_observables", result_dist, obs=sn_observables)
