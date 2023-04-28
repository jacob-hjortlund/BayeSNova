import pytest
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import src.preprocessing as prep

from typing import Tuple

NULL_VALUE = -9999.

@pytest.mark.parametrize(
    "inputs_with_replicas, cfg",
    [
        (
            "inputs_with_replicas",
            ( (10, 3), 4, False, False, False ),
        ),
        (
            "inputs_with_replicas",
            ( (10, 3), 0, False, False, False ),
        ),
        (
            "inputs_with_replicas",
            ( (10, 3), 10, False, False, False ),
        ),
        (
            "inputs_with_replicas",
            ( (10, 3, 3), 4, False, True, False ),
        ),
        (
            "inputs_with_replicas",
            ( (10, 3, 3), 0, False, True, False ),
        ),
        (
            "inputs_with_replicas",
            ( (10, 3, 3), 10, False, True, False ),
        ),
    ], indirect=["inputs_with_replicas"]
)
def test_reorder_duplicates(inputs_with_replicas, cfg):

    (
        inputs, unique_inputs, nonunique_inputs,
        idx_unique_inputs, idx_nonunique_inputs
    ) = inputs_with_replicas(*cfg)

    unique_sn_array, duplicate_sn_array = prep.reorder_duplicates(
        inputs, idx_unique_inputs, idx_nonunique_inputs
    )

    assert np.all(unique_sn_array == unique_inputs)
    for comp, output in zip(nonunique_inputs, duplicate_sn_array):
        assert np.allclose(comp.astype(np.float64), output.astype(np.float64))

@pytest.mark.parametrize(
    "observables_and_covariances, expected_multivariate_gaussians, inputs_config",
    [
        (   
            "observables_and_covariances",
            "expected_multivariate_gaussians",
            ( (10, 3, 3), 4, False, True, False ),
        ),
        (
            "observables_and_covariances",
            "expected_multivariate_gaussians",
            ( (10, 3, 3), 0, False, True, False ),
        ),
        (
            "observables_and_covariances",
            "expected_multivariate_gaussians",
            ( (10, 3, 3), 10, False, True, False ),
        ),
        (
            "observables_and_covariances",
            "expected_multivariate_gaussians",
            ( (10, 3,), 0, True, True, True ),
        )      
    ], indirect=["observables_and_covariances","expected_multivariate_gaussians"]
)
def test_reduced_observables_and_covariances(
    observables_and_covariances, expected_multivariate_gaussians,
    inputs_config
):

    input_observables, input_covariances = observables_and_covariances(*inputs_config)
    means, expected_log_pdfs = expected_multivariate_gaussians(
        input_observables, input_covariances
    )

    reduced_observables, reduced_covariances, reduced_log_factors = prep.reduced_observables_and_covariances(
        input_covariances, input_observables
    )

    output_log_pdfs = np.array(
        [
            stats.multivariate_normal.logpdf(
                obs, mean=mean, cov=cov,
                allow_singular=True
            ) + log_factor
            
            for obs, mean, cov, log_factor in zip(
                reduced_observables, means,
                reduced_covariances, reduced_log_factors
            )
        ]
    ) 
    
    assert np.allclose(expected_log_pdfs, output_log_pdfs)

