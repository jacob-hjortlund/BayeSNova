import pandas as pd
import src.preprocessing as prep
import pytest
import numpy as np


from typing import Tuple

NULL_VALUE = -9999.
rng = np.random.default_rng(seed=42)

def reorder_duplicates(
    sn_array: np.ndarray, idx_unique_sn: np.ndarray,
    idx_duplicate_sn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder duplicate SN arrays to match the order of the unique SN array.

    Args:
        sn_array (np.ndarray): Array of SN properties with shape (N,....)
        idx_unique_sn (np.ndarray): Array of boolean indices of unique SN with shape (N,)
        idx_duplicate_sn (np.ndarray): Array of boolean indices of duplicate SN with shape (M, N)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Reordered unique and duplicate SN arrays. Unique array has
        shape (K, ...) and duplicate array is an object array with shape (M, (L_i, ...))
    """
    
    duplicate_sn_array = []
    for idx in idx_duplicate_sn:
        duplicate_sn_array.append(sn_array[idx])
    duplicate_sn_array = np.array(duplicate_sn_array, dtype=object)

    unique_sn_array = sn_array[idx_unique_sn].copy()

    return unique_sn_array, duplicate_sn_array

def reduced_observables_and_covariances(
    duplicate_covariances: np.ndarray, duplicate_observables: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce the number of observables and covariances by combining
    duplicates.

    Args:
        duplicate_covariances (np.ndarray): Array of duplicate covariance matrices with shape (N,3,3)
        duplicate_observables (np.ndarray): Array of duplicate observable arrays with shape (N,3)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Reduced covariance matrices and observables
    """

    n_duplicates = len(duplicate_covariances)
    reduced_observables = np.zeros((n_duplicates, *duplicate_observables[0].shape[1:]))
    reduced_covariances = np.zeros((n_duplicates, *duplicate_covariances[0].shape[1:]))
    max_value = np.finfo(duplicate_observables[0].dtype).max
    null_cutoff = 10**(np.log10(max_value)/10)

    for i in range(n_duplicates):

        duplicate_obs = duplicate_observables[i][:, :, None]
        duplicate_covs = duplicate_covariances[i]
        duplicate_covs = np.where(
            duplicate_covs == NULL_VALUE, max_value, duplicate_covs
        )
        cov_i_inv = np.linalg.inv(duplicate_covs)
        reduced_cov = np.linalg.inv(
            np.sum(
                cov_i_inv, axis=0
            )
        )

        reduced_obs = np.matmul(
            reduced_cov, np.sum(
                np.matmul(cov_i_inv, duplicate_obs), axis=0
            )
        )

        reduced_cov = np.where(
            np.abs(reduced_cov) > null_cutoff, NULL_VALUE, reduced_cov
        )
        reduced_observables[i] = reduced_obs.squeeze()
        reduced_covariances[i] = reduced_cov
    
    return reduced_observables, reduced_covariances

@pytest.mark.parametrize(
    "inputs_with_replicas",
    [
        ( (10, 3), 4, False ),
        ( (10, 3), 0, False ),
        ( (10, 3), 10, False ),
        ( (10, 3,3), 4, True ),
        ( (10, 3,3), 0, True ),
        ( (10, 3,3), 10, True ),       
    ], indirect=True
)
def test_reorder_duplicates(inputs_with_replicas):

    (
        inputs, unique_inputs, nonunique_inputs,
        idx_unique_inputs, idx_nonunique_inputs
    ) = inputs_with_replicas

    unique_sn_array, duplicate_sn_array = prep.reorder_duplicates(
        inputs, idx_unique_inputs, idx_nonunique_inputs
    )

    assert np.all(unique_sn_array == unique_inputs)
    for comp, output in zip(nonunique_inputs, duplicate_sn_array):
        assert np.allclose(comp.astype(np.float64), output.astype(np.float64))