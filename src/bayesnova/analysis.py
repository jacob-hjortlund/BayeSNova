import numpy as np
import autofit as af
import scipy.special as special

from bayesnova.mixture import TwoPopulationMixture, LogisticLinearWeighting


class Analysis(af.Analysis):
    def __init__(
        self,
        apparent_B_mag: np.ndarray,
        stretch: np.ndarray,
        color: np.ndarray,
        redshift: np.ndarray,
        observed_covariance: np.ndarray,
        calibrator_indeces: np.ndarray = None,
        calibrator_distance_modulus: np.ndarray = None,
        host_properties: np.ndarray = np.zeros((0, 0)),
        host_covariances: np.ndarray = np.zeros((0, 0)),
        volumetric_rate_redshifts=np.zeros((0, 0)),
        volumetric_rate_observations: np.ndarray = np.zeros((0, 0)),
        volumetric_rate_errors: np.ndarray = np.zeros((0, 0)),
        logistic_linear_bounds: tuple = (0.15, 0.85),
        gamma_quantiles_cfg: dict = {
            "lower": 1.0,
            "upper": 20.0,
            "resolution": 0.001,
            "cdf_limit": 0.995,
        },
        **kwargs,
    ):
        self.apparent_B_mag = apparent_B_mag
        self.stretch = stretch
        self.color = color
        self.redshift = redshift
        self.observed_covariance = observed_covariance
        self.calibrator_indeces = calibrator_indeces
        self.calibrator_distance_modulus = calibrator_distance_modulus
        self.host_properties = host_properties
        self.host_covariances = host_covariances
        self.volumetric_rate_redshifts = volumetric_rate_redshifts
        self.volumetric_rate_observations = volumetric_rate_observations
        self.volumetric_rate_errors = volumetric_rate_errors
        self.logistic_linear_bounds = logistic_linear_bounds
        self.kwargs = kwargs

        if self.calibrator_indeces is None:
            self.calibrator_indeces = np.zeros_like(self.apparent_B_mag, dtype=bool)

        if isinstance(gamma_quantiles_cfg, dict):
            self.gamma_quantiles = self.create_gamma_quantiles(**gamma_quantiles_cfg)
        else:
            self.gamma_quantiles = 10.0

    def create_gamma_quantiles(
        self, lower: float, upper: float, resolution: float, cdf_limit: float
    ):
        vals = np.arange(lower, upper, resolution)
        quantiles = np.stack((vals, special.gammaincinv(vals, cdf_limit)))

        return quantiles

    def sn_log_likelihood_function(self, instance) -> float:
        log_likelihoods = instance.log_likelihood(
            apparent_B_mag=self.apparent_B_mag,
            stretch=self.stretch,
            color=self.color,
            redshift=self.redshift,
            observed_covariance=self.observed_covariance,
            calibrator_indeces=self.calibrator_indeces,
            calibrator_distance_modulus=self.calibrator_distance_modulus,
            upper_bound_E_BV=self.gamma_quantiles,
            host_property_observables=self.host_properties,
            host_property_covariance=self.host_covariances,
            **self.kwargs,
        )

        return log_likelihoods

    def host_log_likelihood_function(self, instances: list) -> float:
        host_log_likelihoods = 0.0
        for i, host_model in enumerate(instances):
            host_log_likelihoods += host_model.log_likelihood(
                observations=self.host_properties[:, i],
                variance=self.host_covariances[:, i],
            )

        return host_log_likelihoods

    def log_likelihood_function(self, instance) -> float:
        instance_vars = vars(instance)
        host_model_in_instance = "host_models" in instance_vars
        if host_model_in_instance:
            host_model_in_instance = instance_vars["host_models"] != None
        progenitor_model_in_instance = "progenitor_model" in instance_vars
        is_only_sn = not progenitor_model_in_instance and not host_model_in_instance

        if is_only_sn:
            log_likelihood = self.sn_log_likelihood_function(instance)
        else:
            log_likelihood = self.sn_log_likelihood_function(instance.sn)

        log_likelihood = np.sum(log_likelihood)

        if host_model_in_instance:
            host_log_likelihoods = self.host_log_likelihood_function(
                instances=instance.host_models
            )

            log_likelihood += np.sum(host_log_likelihoods)

        if progenitor_model_in_instance:
            log_likelihood += instance.progenitor_model.log_likelihood(
                volumetric_rate_redshifts=self.volumetric_rate_redshifts,
                volumetric_rate_observations=self.volumetric_rate_observations,
                volumetric_rate_errors=self.volumetric_rate_errors,
            )

        if not np.isfinite(log_likelihood):
            log_likelihood = -1e99

        return log_likelihood


class CepheidAnalysis(af.Analysis):
    def __init__(
        self,
        anchor_distance_modulus: np.ndarray,
        anchor_distance_modulus_error: np.ndarray,
        anchor_parallax: np.ndarray,
        anchor_parallax_error: np.ndarray,
        cepheid_magnitudes: np.ndarray,
        cepheid_magnitude_errors: np.ndarray,
        cepheid_periods: np.ndarray,
        cepheid_metallicities: np.ndarray,
        cepheid_metallicity_errors: np.ndarray = None,
    ):
        self.anchor_distance_modulus = anchor_distance_modulus
        self.anchor_distance_modulus_error = anchor_distance_modulus_error
        self.anchor_parallax = anchor_parallax
        self.anchor_parallax_error = anchor_parallax_error
        self.cepheid_magnitudes = cepheid_magnitudes
        self.cepheid_magnitude_errors = cepheid_magnitude_errors
        self.cepheid_periods = cepheid_periods
        self.cepheid_metallicities = cepheid_metallicities

        if cepheid_metallicity_errors is None:
            cepheid_metallicity_errors = np.zeros_like(cepheid_metallicities)
        self.cepheid_metallicity_errors = cepheid_metallicity_errors

    def anchor_llh(self, instance) -> float:
        anchor_llh = 0.0
        for distmod, distmoderr, anchor_model in zip(
            self.anchor_distance_modulus,
            self.anchor_distance_modulus_error,
            instance.anchor_models,
        ):
            anchor_llh += anchor_model.log_likelihood(
                distance_modulus=distmod, distance_modulus_error=distmoderr
            )

        return anchor_llh

    def cepheid_anchor_distmod_llh(self, instance) -> float:
        anchor_llh = 0.0

        for (
            cepheid_mag,
            cepheid_mag_err,
            cepheid_period,
            cepheid_metallicity,
            cepheid_metallicity_err,
            cepheid_model,
        ) in zip(
            self.cepheid_magnitudes,
            self.cepheid_magnitude_errors,
            self.cepheid_periods,
            self.cepheid_metallicities,
            self.cepheid_metallicity_errors,
            instance.cepheid_anchor_distmod_models,
        ):
            anchor_llh += cepheid_model.log_likelihood(
                mw=cepheid_mag,
                mw_err=cepheid_mag_err,
                period=cepheid_period,
                metallicity=cepheid_metallicity,
                # cepheid_metallicity_err=cepheid_metallicity_err
            )

        return anchor_llh

    def cepheid_parallax_llh(self, instance) -> float:
        cepheid_llh = 0.0
        for (
            anchor_parallax,
            anchor_parallax_err,
            cepheid_mag,
            cepheid_mag_err,
            cepheid_period,
            cepheid_metallicity,
            cepheid_metallicity_err,
            cepheid_model,
        ) in zip(
            self.anchor_parallax,
            self.anchor_parallax_error,
            self.cepheid_magnitudes,
            self.cepheid_magnitude_errors,
            self.cepheid_periods,
            self.cepheid_metallicities,
            self.cepheid_metallicity_errors,
            instance.cepheid_parallax_models,
        ):
            cepheid_llh += cepheid_model.log_likelihood(
                parallax=anchor_parallax,
                parallax_err=anchor_parallax_err,
                mw=cepheid_mag,
                mw_err=cepheid_mag_err,
                period=cepheid_period,
                metallicity=cepheid_metallicity,
                # cepheid_metallicity_err=cepheid_metallicity_err
            )

        return cepheid_llh

    def cepheid_llh(self, instance) -> float:
        cepheid_llh = 0.0
        for (
            cepheid_mag,
            cepheid_mag_err,
            cepheid_period,
            cepheid_metallicity,
            cepheid_metallicity_err,
            cepheid_model,
        ) in zip(
            self.cepheid_magnitudes,
            self.cepheid_magnitude_errors,
            self.cepheid_periods,
            self.cepheid_metallicities,
            self.cepheid_metallicity_errors,
            instance.cepheid_calibrator_models,
        ):
            cepheid_llh += cepheid_model.log_likelihood(
                mw=cepheid_mag,
                mw_err=cepheid_mag_err,
                period=cepheid_period,
                metallicity=cepheid_metallicity,
                # cepheid_metallicity_err=cepheid_metallicity_err
            )

        return cepheid_llh

    def log_likelihood_function(self, instance):
        log_likelihood = 0.0

        if instance.anchor_models is not None:
            log_likelihood += self.anchor_llh(instance)

        if instance.cepheid_anchor_distmod_models is not None:
            log_likelihood += self.cepheid_anchor_distmod_llh(instance)

        if instance.cepheid_parallax_models is not None:
            log_likelihood += self.cepheid_parallax_llh(instance)

        if instance.cepheid_calibrator_models is not None:
            log_likelihood += self.cepheid_llh(instance)

        return log_likelihood
