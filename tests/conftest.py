import os
import pytest
import numpy as np

from typing import Tuple

NULL_VALUE = -9999.

@pytest.fixture
def rng():
    return np.random.default_rng(1337)

@pytest.fixture
def create_random_inputs(rng):
    def _create_random_inputs(
        input_shape: Tuple[int, ...], make_diagonal: bool,
        make_symmetric_and_posdef: bool
    ):
        """
        Generates random inputs.

        Args:
            input_shape (Tuple[int, ...]): Shape of input array
            make_diagonal (bool): Whether to make the inputs diagonal
            make_symmetric_and_posdef (bool): Whether to make the inputs symmetric
            and positive definite

        Returns:
            np.ndarray: Random inputs
        """
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
    def _null_value_insertion(inputs, null_value=NULL_VALUE):
        """
        Inserts null values into inputs.

        Args:
            inputs (np.ndarray): Inputs
            null_value (float): Null value. Defaults to -9999.

        Returns:
            np.ndarray: Inputs with null values
        """

        n = inputs.shape[0]
        #TODO: Generlize to 2d inputs or add check
        n_dimensions = inputs.shape[-1]

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
        Generates inputs with replicas.

        Args:
            input_shape (Tuple[int, ...]): Shape of input array
            n_unique (int): Number of unique inputs
            make_symmetric_and_posdef (bool): Whether to make the inputs symmetric and positive definite
            insert_null_values (bool): Whether to diagonalize and insert null values

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of inputs,
            unique inputs, replica inputs, replica indices, unique indices
        """
        
        n = input_shape[0]
        n_nonunique = n - n_unique

        inputs = create_random_inputs(
            input_shape, make_diagonal=make_diagonal,
            make_symmetric_and_posdef=make_symmetric_and_posdef
        )

        if insert_null_values:
            inputs = null_value_insertion(inputs)

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
