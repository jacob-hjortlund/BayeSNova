import os
import pytest
import numpy as np

from typing import Tuple

NULL_VALUE = -9999.

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def inputs_with_replicas(rng):
    def _inputs_with_replicas(
        input_shape, n_unique, make_symmetric_and_posdef,
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

        if insert_null_values:
            inputs = np.abs(rng.random((n, input_shape[-1])))
            idx_set_null = rng.choice(
                [True, False], size=(n, input_shape[-1])
            )
            inputs[idx_set_null] = NULL_VALUE
            for i in range(n):
                if np.all(inputs[i] == NULL_VALUE):
                    idx = rng.choice(np.arange(input_shape[-1]), 1)[0]
                    inputs[i, idx] = rng.random(1)
            inputs = np.array([np.diag(i) for i in inputs])
        
        else:
            inputs = rng.random(input_shape)

        if make_symmetric_and_posdef and not insert_null_values:
            inputs = inputs + inputs.transpose(0,2,1)
            inputs = np.matmul(inputs, inputs.transpose(0,2,1))

        idx_unique = np.ones(n, dtype=bool)
        idx_unique_indices = np.arange(n)

        min_number_of_replica = 2
        max_number_of_replica = n_nonunique
        possible_no_of_replica = np.arange(min_number_of_replica, max_number_of_replica+1)
        idx_nonunique = np.zeros((0, n), dtype=bool)

        nonunique_inputs = []

        while max_number_of_replica >= min_number_of_replica:

            if len(possible_no_of_replica) > 1:
                possible_no_of_replica = np.delete(possible_no_of_replica, -2)
            chosen_no_of_replica = rng.choice(possible_no_of_replica, 1)[0]

            available_indices = idx_unique_indices[idx_unique]
            indices = rng.choice(available_indices, chosen_no_of_replica, replace=False)

            idx_replicas = np.zeros((1,n), dtype=bool)
            idx_replicas[0, indices] = True
            idx_nonunique = np.concatenate(
                (idx_nonunique, idx_replicas), axis=0
            )

            max_number_of_replica = n_nonunique - np.sum(idx_nonunique)
            possible_no_of_replica = possible_no_of_replica[possible_no_of_replica <= max_number_of_replica]
            idx_unique[indices] = False

            nonunique_inputs.append(inputs[idx_replicas.squeeze()])
        
        nonunique_inputs = np.array(nonunique_inputs, dtype=object)
        unique_inputs = inputs[idx_unique]

        reduced_idx_nonunique = np.any(idx_nonunique, axis=0)
        is_equal_to_n_nonunique = np.sum(idx_nonunique) == n_nonunique
        no_overlap = np.sum(reduced_idx_nonunique & idx_unique) == 0

        if not no_overlap:
            raise ValueError("There is overlap between the unique and non-unique indices.")
        
        if not is_equal_to_n_nonunique:
            error_string = (
                "The number of non-unique indices is not equal to the number of non-unique inputs." +
                f"Number of non-unique indices: {np.sum(idx_nonunique)}" +
                f"Number of non-unique inputs: {n_nonunique}"
            )
            raise ValueError(error_string)
        
        return inputs, unique_inputs, nonunique_inputs, idx_unique, idx_nonunique

    return _inputs_with_replicas
