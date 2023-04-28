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
    "rng, inputs_with_replicas, inputs_config",
    [
        (   
            "rng",
            "inputs_with_replicas",
            ( (10, 3, 3), 4, False, True, False ),
        ),
        (
            "rng",
            "inputs_with_replicas",
            ( (10, 3, 3), 0, False, True, False ),
        ),
        (
            "rng",
            "inputs_with_replicas",
            ( (10, 3, 3), 10, False, True, False ),
        ),
        (
            "rng",
            "inputs_with_replicas",
            ( (10, 3, 3), 0, True, True, True ),
        )      
    ], indirect=["rng","inputs_with_replicas"]
)
def test_reduced_observables_and_covariances(
    rng, inputs_with_replicas, inputs_config
):

    (
        _, _, nonunique_covariances, _, _
    ) = inputs_with_replicas(*inputs_config)
    n_nonunique = len(nonunique_covariances)

    if n_nonunique == 0:
        n_dimensions = 0
    else:
        n_dimensions = nonunique_covariances[0].shape[-1]

    nonunique_observables = []
    for cov in nonunique_covariances:
        idx_null = (
            np.array(
                [np.diag(tmp_cov) for tmp_cov in cov]
            ) == NULL_VALUE
        )
        tmp_obs = rng.normal(size=idx_null.shape)
        tmp_obs[idx_null] = NULL_VALUE
        nonunique_observables.append(tmp_obs)
    nonunique_observables = np.array(nonunique_observables, dtype=object)
    mean = rng.normal(size=(n_dimensions,))

    max_value = np.finfo(np.float64).max
    expected_log_pdf = np.zeros(n_nonunique)
    for i in range(n_nonunique):
        for obs, cov in zip(nonunique_observables[i], nonunique_covariances[i]):

            obs = np.where(
                obs == NULL_VALUE, 0., obs
            )

            cov = np.where(
                cov == NULL_VALUE, max_value, cov
            )

            expected_log_pdf[i] += stats.multivariate_normal.logpdf(
                obs, mean=mean, cov=cov, allow_singular=True
            )

    reduced_observables, reduced_covariances, reduced_log_factors = prep.reduced_observables_and_covariances(
        nonunique_covariances, nonunique_observables
    )

    output_log_pdf = np.zeros(n_nonunique)
    for i in range(n_nonunique):

        obs = np.where(
            reduced_observables[i] == NULL_VALUE, 0., reduced_observables[i]
        )

        cov = np.where(
            reduced_covariances[i] == NULL_VALUE, max_value, reduced_covariances[i]
        )

        output_log_pdf[i] = stats.multivariate_normal.logpdf(
            obs, mean=mean, cov=cov,
            allow_singular=True
        ) + reduced_log_factors[i]
    
    assert np.allclose(expected_log_pdf, output_log_pdf)

