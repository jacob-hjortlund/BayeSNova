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
import src.bayesnova.predictive.single_population as single_pop
import src.bayesnova.utils as utils

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

def multi_pop_sn_model_builder(
    sn_app_magnitudes: np.ndarray,
    sn_stretch: np.ndarray,
    sn_colors: np.ndarray,
    sn_redshifts: np.ndarray,
    sn_covariances: np.ndarray,
    calibrator_indices: np.ndarray = None,
    calibrator_distance_moduli: np.ndarray = None,
    selection_bias_correction: np.ndarray = None,
    sn_model_config: dict = {},
    mixture_model_config: dict = {}
) -> Callable:
    
    raise_error, error_message = check_multi_pop_sn_model_config(
        mixture_model_config=mixture_model_config
    )
    if raise_error:
        raise ValueError(error_message)

    mixture_model_name = mixture_model_config.get("model_name", "Mixture Model")
    n_mixture_components = mixture_model_config.get("n_components", 2)
    shared_parameters = mixture_model_config.get("shared_parameters", [])
    independent_parameters = mixture_model_config.get("independent_parameters", [])
    cosmological_parameters = mixture_model_config.get("cosmological_parameters", [])
    mixture_parameters = mixture_model_config.get("mixture_parameters", [])
    all_free_parameter_names = shared_parameters + independent_parameters + cosmological_parameters
    fixed_sn_parameters = mixture_model_config.get("fixed_sn_parameters", {})
    fixed_mixture_parameters = mixture_model_config.get("fixed_mixture_parameters", {})
    use_physical_mixture_weights = mixture_model_config.get("use_physical_mixture_weights", False)

    n_shared_parameters = len(shared_parameters)
    n_independent_parameters = len(independent_parameters)
    n_cosmological_parameters = len(cosmological_parameters)
    n_total_independent_parameters = n_mixture_components * n_independent_parameters
    n_mixture_parameters = len(mixture_parameters)

    if not use_physical_mixture_weights:
        n_supernoave = sn_app_magnitudes.shape[1]
        mixture_weight_function = lambda mixture_parameters: constant_mixture_weights(
            n_supernovae=n_supernoave, **mixture_parameters
        )
    else:
        print("Using physical mixture weights.")

    sn_model_config = sn_model_config.copy()
    sn_model_config["free_parameters"] = all_free_parameter_names
    sn_model_config['is_component'] = True
    sn_model_config['fixed_parameters'] = fixed_sn_parameters

    sn_models = []
    for i in range(n_mixture_components):
        sn_model_config_i = sn_model_config.copy()
        sn_model_config_i["model_name"] = f"{mixture_model_name} SN_{i}"
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
    
    def multi_pop_sn_model(
        sampled_parameters: np.ndarray,
    ) -> float:

        shared_params = sampled_parameters[:n_shared_parameters]
        cosmological_params = sampled_parameters[
            n_shared_parameters+n_total_independent_parameters:
            n_shared_parameters+n_total_independent_parameters+n_cosmological_parameters
        ]
        mixture_params = sampled_parameters[
            n_shared_parameters+n_total_independent_parameters+n_cosmological_parameters:
            n_shared_parameters+n_total_independent_parameters+n_cosmological_parameters+n_mixture_parameters
        ]

        mixture_params_dict = utils.map_array_to_dict(
            array=mixture_params,
            array_names=mixture_parameters
        )
        mixture_params_dict = mixture_params_dict | fixed_mixture_parameters

        mixture_weights = mixture_weight_function(**mixture_params_dict)

        population_likelihoods = []
        for i in range(n_mixture_components):

            independent_params = sampled_parameters[
                n_shared_parameters+i*n_independent_parameters:
                n_shared_parameters+(i+1)*n_independent_parameters
            ]
            input_params = np.concatenate(
                [shared_params, independent_params, cosmological_params]
            )

            population_likelihood = sn_models[i](input_params) * mixture_weights[i]

            population_likelihoods.append(population_likelihood)
        
        return 1
    
    return multi_pop_sn_model