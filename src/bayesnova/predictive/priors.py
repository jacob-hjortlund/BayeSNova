import numpy as np
import scipy.stats as stats

from typing import Callable

def uniform_log_prior(value: float, lower: float = -np.inf, upper: float = np.inf):
    """
    Uniform log prior.

    Args:
        value (float): Value to evaluate.
        lower (float, optional): Lower bound. Defaults to -np.inf.
        upper (float, optional): Upper bound. Defaults to np.inf.
        
    Returns:
        float: Log prior value.
    """

    if value < lower or value > upper:
        return -np.inf
    else:
        return 0.

def truncnorm_log_prior(
    value: float, lower: float = -np.inf, upper: float = np.inf,
    mu: float = 0., sigma: float = 1.
):
    """
    Truncated normal log prior.

    Args:
        value (float): Value to evaluate.
        lower (float, optional): Lower bound. Defaults to -np.inf.
        upper (float, optional): Upper bound. Defaults to np.inf.
        mu (float, optional): Mean. Defaults to 0.
        sigma (float, optional): Standard deviation. Defaults to 1.
        
    Returns:
        float: Log prior value.
    """

    if value < lower or value > upper:
        return -np.inf
    else:
        return stats.truncnorm.logpdf(
            value, (lower - mu) / sigma, (upper - mu) / sigma,
            loc=mu, scale=sigma
        )

def power(x, base=10):
    """
    Power function.

    Args:
        x (float): Value to evaluate.
        base (int, optional): Base. Defaults to 10.
        
    Returns:
        float: Power value.
    """

    return base ** x

def sampling_transform(parameter_dict: dict, transform_config):
    """
    Applies sampling transforms to parameters.

    Args:
        parameter_dict (dict): Parameter dictionary.
        transform_config (dict): Transform config.

    Returns:
        dict: Transformed parameter dictionary.
    """

    param_dict = parameter_dict.copy()
    params_with_transforms = set(param_dict.keys()).intersection(transform_config.keys())
    for parameter in params_with_transforms:
        
        sampling_transform_name = transform_config[parameter].get('transform', None)
        sampling_transform_kwargs = transform_config[parameter].get('kwargs', {})
        
        if sampling_transform_name is not None:
            sampling_transform = globals()[sampling_transform_name]
        else:
            sampling_transform = lambda x, **kwargs: x
        
        param_dict[parameter] = sampling_transform(
            param_dict[parameter], **sampling_transform_kwargs
        )
    
    return param_dict



def ordering_prior_builder(
    independent_parameters_to_compare: list,
) -> Callable:
    """
    Builds an ordering prior function.

    Args:
        independent_parameters_to_compare (list): List of independent parameters to compare.

    Returns:
        Callable: Ordering prior function.
    """
    
    def check_if_ordered(parameter_dict: dict) -> float:
        """
        Checks if array of each independent population parameter is ordered.

        Args:
            parameter_dict (dict): Dictionary of parameter values. Keys are parameter names
            and values are arrays of population parameter values.

        Returns:
            float: Log prior value, -inf if not ordered, 0 otherwise. 
        """

        prior_value = 0.
        parameters_to_compare = set(parameter_dict.keys()).intersection(independent_parameters_to_compare)
        for parameter in parameters_to_compare:
            values_to_compare = parameter_dict[parameter]
            is_ordered = np.all(values_to_compare[:-1] < values_to_compare[1:])
            if not is_ordered:
                prior_value = -np.inf
                break

        return prior_value

    return check_if_ordered

def prior_builder(
    model_name: str,
    free_parameters: list[str] = [],
    prior_config: dict = {},
) -> Callable:
    """
    Builds a prior function from a model config.

    Args:
        model_name (str): Model name.
        free_parameters (list[str], optional): List of free parameters. Defaults to [].
        prior_config (dict, optional): Prior configuration. Defaults to {}.

    Returns:
        Callable: Prior function.
    """

    parameters_in_prior = set(prior_config.keys()).intersection(free_parameters)

    def prior_func(
        parameter_dict: dict
    ) -> float:
        """
        Prior function.

        Args:
            parameter_dict (dict): Dictionary of parameter values.

        Returns:
            float: Log prior value.
        """

        prior_value = 0.
        for parameter in parameters_in_prior:
            
            is_gaussian = (
                "mean" in prior_config[parameter].keys() and
                "sigma" in prior_config[parameter].keys()
            )
            is_uniform = (
                not is_gaussian and
                (
                    "lower" in prior_config[parameter].keys() or
                    "upper" in prior_config[parameter].keys()
                )
            )
            prior_is_not_valid = not is_gaussian and not is_uniform

            if prior_is_not_valid:
                raise ValueError(f'Prior for parameter {parameter} in {model_name} model is not valid.')

            lower_bound = prior_config[parameter].get('lower', -np.inf)
            upper_bound = prior_config[parameter].get('upper', np.inf)

            if is_gaussian:
                
                prior_value += truncnorm_log_prior(
                    parameter_dict[parameter],
                    lower=lower_bound,
                    upper=upper_bound,
                    mu=prior_config[parameter]['mean'],
                    sigma=prior_config[parameter]['sigma']
                )

            if is_uniform:

                prior_value += uniform_log_prior(
                    parameter_dict[parameter],
                    lower=lower_bound,
                    upper=upper_bound
                )

            if prior_value == -np.inf:
                break

        return prior_value
    
    return prior_func
