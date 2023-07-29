import numpy as np
import autofit as af

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
    ):
        
        self.apparent_B_mag = apparent_B_mag
        self.stretch = stretch
        self.color = color
        self.redshift = redshift
        self.observed_covariance = observed_covariance
        self.calibrator_indeces = calibrator_indeces
        self.calibrator_distance_modulus = calibrator_distance_modulus

        if self.calibrator_indeces is None:
            self.calibrator_indeces = np.zeros_like(self.apparent_B_mag, dtype=bool)

    def log_likelihood_function(self, instance) -> float:

        log_likelihoods = instance.log_likelihood(
            apparent_B_mag=self.apparent_B_mag,
            stretch=self.stretch,
            color=self.color,
            redshift=self.redshift,
            observed_covariance=self.observed_covariance,
            calibrator_indeces=self.calibrator_indeces,
            calibrator_distance_modulus=self.calibrator_distance_modulus,
        )

        if np.any(np.isfinite(log_likelihoods) == False):
            log_likelihood = np.finfo(np.float64).min
        else:
            log_likelihood = np.sum(log_likelihoods)

        return log_likelihood
