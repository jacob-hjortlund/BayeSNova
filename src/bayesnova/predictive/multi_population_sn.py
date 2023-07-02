import inspect
import warnings
import numpy as np
import numba as nb
import NumbaQuadpack as nq
import bayesnova.utils as utils
import scipy.stats as stats
import scipy.special as special
import astropy.cosmology as cosmo

from functools import partial
from astropy.units import Gyr
from typing import Callable

import bayesnova.preprocessing as prep
import bayesnova.cosmology_utils as cosmo_utils
import src.bayesnova.predictive.priors as priors
import src.bayesnova.predictive.single_population_sn as single_pop
import src.bayesnova.utils as utils
from src.bayesnova.cosmology.volumetric_rates import volumetric_rates

# --------------------------------------- MULTIPLE POPULATION SUPERNOVA MODEL ---------------------------------------

# TODO: ADD CHECKS FOR FIXED MIXTURE PARAMETERS
def check_multi_pop_sn_model_config(
    mixture_model_config: dict
) -> tuple[bool, str]:
    """Check that the mixture model config is valid.

    Args:
        mixture_model_config (dict): The mixture model config.

    Returns:
        tuple[bool, str]: Whether the config is valid and an error message if not.
    """
    
    n_mixture_components = mixture_model_config.get("n_components", 2)
    mixture_parameters = mixture_model_config.get("mixture_parameters", [])
    use_physical_mixture_weights = mixture_model_config.get("use_physical_mixture_weights", False)

    is_two_population_model = n_mixture_components == 2
    mixture_parameters_match_components = len(mixture_parameters) == n_mixture_components - 1

    if n_mixture_components == 1:
        error_message = "Mixture model must have more than one component."
        return True, error_message

    if not mixture_parameters_match_components and not use_physical_mixture_weights:
        error_message = (
            "When not using physical mixture weights, the number of mixture parameters" +
            " must match the number of components minus one. Current no. of components is {n_mixture_components}" +
            " and no. of mixture parameters is {n_mixture_parameters}."
        )
        return True, error_message
    
    if use_physical_mixture_weights and not is_two_population_model:
        error_message = (
            "When using physical mixture weights, the number of components must be two." +
            " Current no. of components is {n_mixture_components}."
        )
        return True, error_message
    
    return False, None

def constant_mixture_weights(
    n_supernovae: int,
    **kwargs
) -> np.ndarray:
    """Create a constant mixture weight array. Takes input as dictionary via kwargs
    for consistency with other mixture weight functions.

    Args:
        n_supernovae (int): The number of supernovae.
        kwargs (dict): Mixture parameter dictionary.
    Returns:
        np.ndarray: The mixture weights.
    """
    
    mixture_parameters, _ = utils.map_dict_to_array(dict=kwargs)

    mixture_weights = [
        np.ones(n_supernovae) * mixture_parameter for mixture_parameter in mixture_parameters
    ]
    mixture_weights += [np.ones(n_supernovae) * (1 - np.sum(mixture_parameters))]
    mixture_weights = np.row_stack(mixture_weights)

    return mixture_weights

def redshift_dependent_mixture_weights(
    redshifts: np.ndarray, H0: float, Om0: float,
    w0: float, wa: float, eta: float, 
    prompt_fraction: float, **kwargs
) -> np.ndarray:
    """Create a redshift dependent mixture weight array based on volumetric SN Ia rates.

    Args:
        redshifts (np.ndarray): The redshifts.
        H0 (float): The Hubble constant.
        Om0 (float): The matter density.
        w0 (float): The dark energy equation of state parameter.
        wa (float): The dark energy equation of state evolution parameter.
        eta (float): The SN Ia delay time distribution normalization.
        prompt_fraction (float): The prompt progenitor channel fraction.

    Returns:
        np.ndarray: The mixture weights.
    """

    sn_ia_rates = volumetric_rates(
        z=redshifts, H0=H0, Om0=Om0, w0=w0, wa=wa,
        eta=eta, prompt_fraction=prompt_fraction
        **kwargs
    )
    valid_rates = np.all(np.isfinite(sn_ia_rates))

    if valid_rates:
        prompt_population_weight = sn_ia_rates[:, 1] / sn_ia_rates[:, 0]
        delayed_population_weight = 1 - prompt_population_weight
        mixture_weights = np.row_stack([prompt_population_weight, delayed_population_weight])
    else:
        mixture_weights = np.ones((2, len(redshifts))) * -np.inf

    return mixture_weights

# TODO: CONSIDER HOW TO USE PRIOR FUNC + COMPARING STRETCH IN PRIOR
def multi_pop_sn_model_builder(
    sn_app_magnitudes: np.ndarray,
    sn_stretch: np.ndarray,
    sn_colors: np.ndarray,
    sn_redshifts: np.ndarray,
    sn_covariances: np.ndarray,
    calibrator_indices: np.ndarray = None,
    calibrator_distance_moduli: np.ndarray = None,
    selection_bias_correction: np.ndarray = None,
    mixture_model_config: dict = {}
) -> Callable:
    
    # Check Mixture Model Config
    raise_error, error_message = check_multi_pop_sn_model_config(
        mixture_model_config=mixture_model_config
    )
    if raise_error:
        raise ValueError(error_message)

    sn_model_config = mixture_model_config.get("single_pop", {})
    mixture_model_config = mixture_model_config.get("multi_pop", {})

    # Mixture Settings
    mixture_model_name = mixture_model_config.get("model_name", "Mixture Model")
    n_mixture_components = mixture_model_config.get("n_components", 2)
    mixture_parameters = mixture_model_config.get("mixture_parameters", [])
    fixed_mixture_parameters = mixture_model_config.get("fixed_mixture_parameters", {})
    mixture_kwargs = mixture_model_config.get("mixture_kwargs", {})
    use_physical_mixture_weights = mixture_model_config.get("use_physical_mixture_weights", False)
    mixture_prior_config = mixture_model_config.get("priors", {})

    # SN Model Settings
    shared_sn_parameters = mixture_model_config.get("shared_parameters", [])
    independent_sn_parameters = mixture_model_config.get("independent_parameters", [])
    fixed_sn_parameters = mixture_model_config.get("fixed_sn_parameters", {})
    independent_parameters_to_compare = mixture_model_config.get("independent_parameters_to_compare", [])

    # Cosmology Settings
    cosmological_parameters = mixture_model_config.get("cosmological_parameters", [])
    fixed_cosmological_parameters = mixture_model_config.get("fixed_cosmological_parameters", {})
    
    all_free_sn_parameter_names = shared_sn_parameters + independent_sn_parameters + cosmological_parameters

    n_sne = sn_app_magnitudes.shape[1]
    n_shared_parameters = len(shared_sn_parameters)
    n_independent_parameters = len(independent_sn_parameters)
    n_cosmological_parameters = len(cosmological_parameters)
    n_total_independent_parameters = n_mixture_components * n_independent_parameters
    n_mixture_parameters = len(mixture_parameters)

    # Setup Mixture Weight Function
    if not use_physical_mixture_weights:
        mixture_weight_function = lambda mixture_parameters: constant_mixture_weights(
            n_supernovae=n_sne, **mixture_parameters
        )
    else:
        mixture_kwargs = mixture_kwargs | fixed_cosmological_parameters
        mixture_weight_function = lambda mixture_parameters: redshift_dependent_mixture_weights(
            redshifts=sn_redshifts, **mixture_parameters
        )

    # Setup single population SN Models
    sn_model_config = sn_model_config.copy()
    sn_model_config["free_parameters"] = all_free_sn_parameter_names
    sn_model_config['is_component'] = True
    sn_model_config['fixed_parameters'] = fixed_sn_parameters
    sn_model_config['priors'] = sn_model_config.get('priors', {}) | mixture_prior_config

    sn_models = []
    for i in range(n_mixture_components):
        sn_model_config_i = sn_model_config.copy()
        sn_model_config_i["model_name"] = f"{mixture_model_name} Pop {i}"
        sn_models.append(
            single_pop.single_pop_sn_model_builder(
                sn_app_magnitudes=sn_app_magnitudes,
                sn_stretch=sn_stretch,
                sn_colors=sn_colors,
                sn_redshifts=sn_redshifts,
                sn_covariances=sn_covariances,
                calibrator_indices=calibrator_indices,
                calibrator_distance_moduli=calibrator_distance_moduli,
                selection_bias_correction=selection_bias_correction,
                sn_model_config=sn_model_config_i
            )
        )
    
    # Setup Priors
    ordering_prior_function = priors.ordering_prior_builder(
        independent_parameters_to_compare=independent_parameters_to_compare
    )
    prior_function = priors.prior_builder(
        model_name=mixture_model_name,
        free_parameters=mixture_parameters,
        prior_config=mixture_prior_config
    )

    def multi_pop_sn_model(
        sampled_parameters: np.ndarray,
    ) -> float:

        # SNe Model Parameters
        shared_params = sampled_parameters[:n_shared_parameters]
        independent_params = np.split(
            sampled_parameters[
                n_shared_parameters:n_shared_parameters + n_total_independent_parameters
            ], n_mixture_components
        )

        independent_params_dict = utils.map_array_to_dict(
            array=np.array(independent_params).T,
            array_names=independent_sn_parameters
        )

        logprior = ordering_prior_function(independent_params_dict)
        if np.isinf(logprior):
            return logprior

        # Cosmological Parameters
        cosmological_params = sampled_parameters[
            n_shared_parameters + n_total_independent_parameters:
            n_shared_parameters + n_total_independent_parameters + n_cosmological_parameters
        ]

        cosmological_params_dict = utils.map_array_to_dict(
            array=cosmological_params,
            array_names=cosmological_parameters
        )

        # Mixture Parameters
        mixture_params = sampled_parameters[
            n_shared_parameters + n_total_independent_parameters + n_cosmological_parameters:
            n_shared_parameters + n_total_independent_parameters + n_cosmological_parameters + n_mixture_parameters
        ]

        mixture_params_dict = utils.map_array_to_dict(
            array=mixture_params,
            array_names=mixture_parameters
        )

        logprior += prior_function(mixture_params_dict | cosmological_params_dict)
        if np.isinf(logprior):
            return logprior

        mixture_inputs_dict = mixture_params_dict | fixed_mixture_parameters | mixture_kwargs
        mixture_weights = mixture_weight_function(**mixture_inputs_dict)
        
        valid_mixture_weights = np.all(np.isfinite(mixture_weights))
        if not valid_mixture_weights:
            return -np.inf
        
        log_mixture_weights = np.log(mixture_weights)

        population_log_likelihoods = np.zeros((n_sne, n_mixture_components))
        for i in range(n_mixture_components):

            independent_params_i = independent_params[i]
            input_params = np.concatenate(
                [shared_params, independent_params_i, cosmological_params]
            )

            population_log_likelihood = sn_models[i](input_params) + log_mixture_weights[i]
            if np.any(np.isnan(population_log_likelihood)):
                return -np.inf

            population_log_likelihoods[:, i] = population_log_likelihood
        
        sn_probs = special.logsumexp(population_log_likelihoods, axis=1)
        total_log_likelihood = np.sum(sn_probs)

        return total_log_likelihood
    
    return multi_pop_sn_model