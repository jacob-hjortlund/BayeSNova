import os
import pytest
import numpy as np
import scipy.integrate as integrate
import scipy.special as special

from typing import Tuple

NULL_VALUE = -9999.
MAX_VALUE = np.finfo(np.float64).max

def multivariate_gaussian_logpdf(
    x: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> float:
    """
    Multivariate Gaussian log-pdf.

    Args:
        x (np.ndarray): Input array with shape (n_dimensions,)
        mean (np.ndarray): Mean vector with shape (n_dimensions,)
        cov (np.ndarray): Covariance matrix with shape (n_dimensions, n_dimensions)

    Returns:
        logpdf (float): Log-pdf
    """
    
    x = x.astype(np.float64)
    mean = mean.astype(np.float64)
    cov = cov.astype(np.float64)

    n_dimensions = x.shape[-1]
    cov_inv = np.linalg.inv(cov)
    
    logpdf = -0.5 * (
        n_dimensions * np.log(2 * np.pi) + 
        np.linalg.slogdet(cov)[1] +
        (x-mean).T @ cov_inv @ (x-mean)
    )

    return logpdf

def gamma_logpdf(x: np.ndarray, a: float, b: float = 1) -> float:
    """
    Gamma log-pdf.

    Args:
        x (np.ndarray): Input array with shape (n_dimensions,)
        a (float): Shape parameter
        b (float): Rate parameter. Defaults to 1.

    Returns:
        logpdf (float): Log-pdf
    """

    logpdf = (a-1) * np.log(x) - b * x - a * np.log(b) - special.gammaln(a)

    return logpdf

@pytest.fixture
def rng():
    """
    Random number generator.

    Returns:
        np.random.Generator: Random number generator
    """

    return np.random.default_rng(1337)

@pytest.fixture
def dict2class():
    """
    Converts dictionary to class.

    Returns:
        Dict2Class: Class
    """

    class Dict2Class(object):
        def __init__(self, dictionary):
            for key in dictionary:
                setattr(self, key, dictionary[key])

    return Dict2Class

@pytest.fixture
def create_random_inputs(rng):

    def _create_random_inputs(
        input_shape: Tuple[int, ...], make_diagonal: bool,
        make_symmetric_and_posdef: bool
    ) -> np.ndarray:
        """
        Generates random inputs.

        Args:
            input_shape (Tuple[int, ...]): Shape of input array
            make_diagonal (bool): Whether to make the inputs diagonal. Requires that
            len(input_shape) == 2.
            make_symmetric_and_posdef (bool): Whether to make the inputs symmetric
            and positive definite

        Returns:
            np.ndarray: Random inputs
        """
        #TODO: Add check for diagonal and shape match
        inputs = rng.random(input_shape)
        if make_diagonal:
            inputs = np.array([np.diag(i) for i in inputs])
        if make_symmetric_and_posdef:
            inputs = inputs + inputs.transpose(0,2,1)
            inputs = np.matmul(inputs, inputs.transpose(0,2,1))
        
        return inputs

    return _create_random_inputs

@pytest.fixture
def null_value_insertion(rng):

    def _null_value_insertion(inputs, keep_diagonal, null_value=NULL_VALUE):
        """
        Inserts null values into inputs.

        Args:
            inputs (np.ndarray): Inputs
            keep_diagonal (bool): Whether to only insert null values on the diagonal
            null_value (float): Null value. Defaults to -9999.

        Returns:
            np.ndarray: Inputs with null values
        """

        n = inputs.shape[0]
        #TODO: Generlize to 2d inputs or add check
        n_dimensions = inputs.shape[-1]

        if keep_diagonal:
            inputs = np.array([np.diag(i) for i in inputs])

        idx_set_null = rng.choice(
            [True, False], size=(n, n_dimensions)
        )
        inputs[idx_set_null] = null_value

        for i in range(n):
            all_values_are_null = np.all(inputs[i] == null_value)
            if all_values_are_null:
                row_indeces = np.arange(n_dimensions)
                idx = rng.choice(row_indeces, 1)[0]
                inputs[i, idx] = rng.random(1)

        no_values_are_null = np.all(inputs != null_value)
        if no_values_are_null:
            for i in range(n):
                row_indeces = np.arange(n_dimensions)
                idx = rng.choice(row_indeces, 1)[0]
                inputs[i, idx] = null_value

        if keep_diagonal:
            inputs = np.array([np.diag(i) for i in inputs])

        return inputs
    
    return _null_value_insertion

@pytest.fixture
def sample_nonunique_inputs(rng):

    def _sample_non_unique_inputs(
        inputs, available_indices, max_number_of_replica,
        min_number_of_replica 
    ):
        """
        Samples non-unique inputs.

        Args:
            inputs (np.ndarray): Inputs
            available_indices (np.ndarray): Available indices
            max_number_of_replica (int): Maximum number of replicas
            min_number_of_replica (int): Minimum number of replicas

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of non-unique inputs,
            boolean replica indices
        """
        n_inputs = inputs.shape[0]

        possible_no_of_replica = np.arange(min_number_of_replica, max_number_of_replica+1)
        if max_number_of_replica != min_number_of_replica:
            possible_no_of_replica = np.delete(possible_no_of_replica, -2)
        chosen_no_of_replica = rng.choice(possible_no_of_replica, 1)[0]

        indices = rng.choice(available_indices, chosen_no_of_replica, replace=False)
        idx_replicas = np.zeros((1,n_inputs), dtype=bool)
        idx_replicas[0, indices] = True
        nonunique_inputs = inputs[idx_replicas.squeeze()]

        return nonunique_inputs, idx_replicas
    
    return _sample_non_unique_inputs

@pytest.fixture
def check_generated_input():

    def _check_generated_input(
        idx_nonunique, idx_unique, n_total, n_unique
    ):
        """
        Checks generated input.

        Args:
            idx_nonunique (np.ndarray): Boolean array of non-unique indices
            idx_unique (np.ndarray): Boolean array of unique indices
            n_total (int): Total number of inputs
            n_unique (int): Number of unique inputs
        """
        n_nonunique = n_total - n_unique

        not_equal_to_total = (np.sum(idx_nonunique) + np.sum(idx_unique)) != n_total
        n_nonunique_incorrect = np.sum(idx_nonunique) != n_nonunique
        n_unique_incorrect = np.sum(idx_unique) != n_unique
        overlap_between_unique_nonunique = np.sum(idx_nonunique & idx_unique) != 0

        if n_nonunique_incorrect:
            raise ValueError(
                f"Number of non-unique inputs is incorrect. Expected {n_nonunique}, got {np.sum(idx_nonunique)}."
            )
        if n_unique_incorrect:
            raise ValueError(
                f"Number of unique inputs is incorrect. Expected {n_unique}, got {np.sum(idx_unique)}."
            )
        if overlap_between_unique_nonunique:
            raise ValueError(
                "Overlap between unique and non-unique inputs."
            )
        if not_equal_to_total:
            raise ValueError(
                "Number of unique and non-unique inputs is not equal to total number of inputs."
            )
    
    return _check_generated_input

@pytest.fixture
def inputs_with_replicas(
    create_random_inputs, null_value_insertion,
    sample_nonunique_inputs, check_generated_input
):
    
    def _inputs_with_replicas(
        input_shape, n_unique, make_diagonal,
        make_symmetric_and_posdef,
        insert_null_values
    ):
        """
        Creates inputs with replicas.

        Args:
            input_shape (Tuple[int, ...]): Input shape
            n_unique (int): Number of unique inputs
            make_diagonal (bool): Make diagonal
            make_symmetric_and_posdef (bool): Make symmetric and positive definite
            insert_null_values (bool): Insert null values

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of inputs and boolean replica indices
        """
        
        n = input_shape[0]
        n_nonunique = n - n_unique

        inputs = create_random_inputs(
            input_shape, make_diagonal=make_diagonal,
            make_symmetric_and_posdef=make_symmetric_and_posdef
        )

        if insert_null_values:
            inputs = null_value_insertion(inputs, keep_diagonal=make_diagonal)

        idx_unique = np.ones(n, dtype=bool)
        idx_unique_indices = np.arange(n)
        
        min_number_of_replica = 2
        idx_nonunique = np.zeros((0, n), dtype=bool)
        nonunique_inputs = []

        while n_nonunique >= min_number_of_replica:

            nonunique_input_i, idx_replicas_i = sample_nonunique_inputs(
                inputs, idx_unique_indices[idx_unique], n_nonunique, min_number_of_replica
            )
            idx_nonunique = np.concatenate(
                (idx_nonunique, idx_replicas_i), axis=0
            )
            idx_unique[idx_replicas_i.squeeze()] = False
            nonunique_inputs.append(nonunique_input_i)
            n_nonunique = n_nonunique - np.sum(idx_replicas_i)
        
        nonunique_inputs = np.array(nonunique_inputs, dtype=object)
        unique_inputs = inputs[idx_unique]

        check_generated_input(
            idx_nonunique, idx_unique, n, n_unique
        )
        
        return inputs, unique_inputs, nonunique_inputs, idx_unique, idx_nonunique

    return _inputs_with_replicas

@pytest.fixture
def create_random_observables(rng):

    def _create_random_observables(covariances, null_value=NULL_VALUE):
        """
        Creates random observables.

        Args:
            covariances (np.ndarray): Covariances with shape (n_nonunique, n_replicas, n_dim, n_dim)
            null_value (float, optional): Null value. Defaults to NULL_VALUE.

        Returns:
            np.ndarray: Observables with shape (n_nonunique, n_replicas, n_dim)
        """

        n = covariances.shape[0]
        null_values_present = np.any(covariances == null_value)
        if null_values_present:
            observables = np.empty(n, dtype=object)
            for i, cov in enumerate(covariances):

                idx_null_values = np.array(
                    [
                        np.diag(cov_i) == null_value for cov_i in cov
                    ]
                )

                obs_i = rng.normal(size=idx_null_values.shape)
                obs_i[idx_null_values] = null_value
                observables[i] = obs_i
        else:
            observables = rng.normal(size=covariances.shape[:-1])
        
        return observables
    
    return _create_random_observables

@pytest.fixture
def observables_and_covariances(
    inputs_with_replicas, create_random_observables,
):
    
    def _observables_and_covariances(
        covariance_shape, n_unique,
        make_diagonal, make_symmetric_and_posdef,
        insert_null_values
    ):
        """
        Generates observables and covariances.

        Args:
            input_shape (Tuple[int, ...]): Shape of input array
            n_unique (int): Number of unique inputs
            n_observables (int): Number of observables
            n_replicas (int): Number of replicas
            make_diagonal (bool): Whether to make the inputs diagonal
            make_symmetric_and_posdef (bool): Whether to make the inputs symmetric and positive definite
            insert_null_values (bool): Whether to diagonalize and insert null values

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of observables, covariances
        """
        _, covariances, _, _, _ = inputs_with_replicas(
            covariance_shape, n_unique, make_diagonal,
            make_symmetric_and_posdef, insert_null_values
        )

        observables = create_random_observables(covariances)

        return observables, covariances
    
    return _observables_and_covariances

@pytest.fixture
def expected_multivariate_gaussians(rng):

    def _expected_multivariate_gaussians(
        observables, covariances, null_value=NULL_VALUE,
        max_value=MAX_VALUE
    ):
        """
        Calculates expected multivariate Gaussian log probabilities.

        Args:
            observables (np.ndarray): Observables with shape (n_nonunique, n_replicas, n_dim)
            covariances (np.ndarray): Covariances with shape (n_nonunique, n_replicas, n_dim, n_dim)
            null_value (float, optional): Null value. Defaults to NULL_VALUE.
            max_value (float, optional): Max value. Defaults to MAX_VALUE.

        Returns:
            np.ndarray: Expected multivariate Gaussian log probabilities with shape (n_nonunique,)
        """
        
        n = observables.shape[0]
        if n == 0:
            n_dimensions = 0
        else:
            n_dimensions = observables[0].shape[1]

        for i in range(n):
            observables[i] = np.where(
                observables[i] == null_value, 0., observables[i]
            )
            covariances[i] = np.where(
                covariances[i] == null_value, max_value, covariances[i]
            )
    
        means = rng.normal(size=(n, n_dimensions))

        expected_log_probs = np.array(
            [
                np.sum(
                    [
                        multivariate_gaussian_logpdf(
                            obs_i, mean=mean, cov=cov_i
                        ) for obs_i, cov_i in zip(obs, cov)
                    ]
                ) for obs, mean, cov in zip(observables, means, covariances)
            ]
        )

        return means, expected_log_probs
    
    return _expected_multivariate_gaussians

@pytest.fixture
def expected_mvg_gamma_convolution():

    def _expected_mvg_gamma_convolution(
        observables, covariances, Rb, sig_Rb,
        tau_Ebv, gamma_Ebv, upper_bound
    ):
        """
        Calculates expected multivariate Gaussian-Gamma convolution log probabilities.

        Args:
            observables (np.ndarray): Observables with shape (n, n_dim)
            covariances (np.ndarray): Covariances with shape (n, n_dim, n_dim)
            Rb (float): Rb
            sig_Rb (float): sig_Rb
            tau_Ebv (float): tau_Ebv
            gamma_Ebv (float): gamma_Ebv
            upper_bound (float): Upper bound

        Returns:
            np.ndarray: Expected multivariate Gaussian-Gamma convolution log probabilities with shape (n,)
        """
        
        def integral(
            x, obs, mean, cov, Rb, sig_Rb, tau_Ebv, gamma_Ebv
        ):  
            
            mean[0] += Rb * tau_Ebv * x
            mean[2] += tau_Ebv * x
            cov[0, 0] += tau_Ebv**2 * sig_Rb**2 * x**2

            mvg_log_pdf = multivariate_gaussian_logpdf(
                obs, mean=mean, cov=cov
            )
            gamma_log_pdf = gamma_logpdf(
                x, gamma_Ebv
            )

            return np.exp(mvg_log_pdf + gamma_log_pdf)
        
        if len(covariances.shape) > 2:
            n = covariances.shape[0]
        else:
            n = 1
            covariances = np.expand_dims(covariances, axis=0)
            observables = np.expand_dims(observables, axis=0)
        n_dimensions = covariances.shape[-1]

        means = np.zeros(n_dimensions)
        outputs = np.zeros(n)

        for i in range(n):
            outputs[i], _ = integrate.quad(
                integral, 0., upper_bound, args=(
                    observables[i], means, covariances[i],
                    Rb, sig_Rb, tau_Ebv, gamma_Ebv
                )
            )
        
        return outputs
    
    return _expected_mvg_gamma_convolution
        
