import numpy as np
import autofit as af
from mixture import TwoPopulationMixture, LogisticLinearWeighting

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
        logistic_linear_bounds: tuple = (0.15, 0.85),
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
        self.logistic_linear_bounds = logistic_linear_bounds

        if self.calibrator_indeces is None:
            self.calibrator_indeces = np.zeros_like(self.apparent_B_mag, dtype=bool)

    def log_likelihood_function(self, instance) -> float:

        instance_vars = vars(instance)
        host_model_in_instance = "host_models" in instance_vars

        if not host_model_in_instance:
            log_likelihoods = instance.log_likelihood(
                apparent_B_mag=self.apparent_B_mag,
                stretch=self.stretch,
                color=self.color,
                redshift=self.redshift,
                observed_covariance=self.observed_covariance,
                calibrator_indeces=self.calibrator_indeces,
                calibrator_distance_modulus=self.calibrator_distance_modulus,
            )
        else:
            sn_log_likelihoods = instance.sn.log_likelihood(
                apparent_B_mag=self.apparent_B_mag,
                stretch=self.stretch,
                color=self.color,
                redshift=self.redshift,
                observed_covariance=self.observed_covariance,
                calibrator_indeces=self.calibrator_indeces,
                calibrator_distance_modulus=self.calibrator_distance_modulus,
            )
            
            sn_weights = instance.sn.weighting_model.calculate_weight(redshift=self.redshift)

            host_log_likelihoods = np.zeros_like(sn_log_likelihoods)
            for i, host_model in enumerate(instance.host_models):
                
                idx_obs = self.host_covariances[:,i] < 1e150
                observations = self.host_properties[idx_obs,i]
                variance = self.host_covariances[idx_obs,i]
                weights = sn_weights[idx_obs]

                if isinstance(host_model, TwoPopulationMixture):
                    
                    if isinstance(host_model.weighting_model, LogisticLinearWeighting):
                    
                        host_weights = host_model.weighting_model.calculate_weight(weights=weights)
                        prior_assertion = np.all(
                            (host_weights > self.logistic_linear_bounds[0]) &
                            (host_weights < self.logistic_linear_bounds[1])
                        )

                        if not prior_assertion:
                            return -1e99

                host_log_likelihoods += host_model.log_likelihood(
                    observations=observations,
                    variance=variance,
                    weights=weights
                )
            
            log_likelihoods = sn_log_likelihoods + host_log_likelihoods

        if np.any(np.isfinite(log_likelihoods) == False):
            log_likelihood = -1e99
        else:
            log_likelihood = np.sum(log_likelihoods)

        return log_likelihood
