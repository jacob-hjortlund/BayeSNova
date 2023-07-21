import numpy as np
import numba as nb
import scipy.stats as stats
import scipy.special as special
import astropy.cosmology as cosmo

from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags, ldqag

NULL_VALUE = -9999.0
H0_CONVERSION_FACTOR = 0.001022
DH_70 = 4282.7494
SPEED_OF_LIGHT = 299792.458 # km/s

class TrippCalibration():

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        w0: float = -1.0,
        wa: float = 0.0,
        M_int: float = -19.3,
        sigma_M_int: float = 0.1,
        alpha: float = 0.141,
        stretch_int: float = 1.0,
        sigma_stretch_int: float = 0.0,
        beta: float = 3.101,
        color_int: float = -0.1,
        sigma_color_int: float = 0.0,
        peculiar_velocity_dispersion: float = 200.0,
    ):
        
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = w0
        self.wa = wa
        self.M_int = M_int
        self.sigma_M_int = sigma_M_int
        self.alpha = alpha
        self.stretch_int = stretch_int
        self.sigma_stretch_int = sigma_stretch_int
        self.beta = beta
        self.color_int = color_int
        self.sigma_color_int = sigma_color_int
        self.peculiar_velocity_dispersion = peculiar_velocity_dispersion

        self.cosmo = cosmo.Flatw0waCDM(H0=H0, Om0=Om0, w0=w0)
    
    def residuals(
        self,
        apparent_B_mag: np.ndarray,
        stretch: np.ndarray,
        color: np.ndarray,
        redshift: np.ndarray,
        calibrator_indeces: np.ndarray,
        calibrator_distance_modulus: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the residuals for the Tripp calibration for given observed SNe.

        Args:
            apparent_B_mag (np.ndarray): The apparent B-band magnitudes.
            stretch (np.ndarray): The stretch of the SNe.
            color (np.ndarray): The color of the SNe.
            redshift (np.ndarray): The redshift of the SNe.
            calibrator_indeces (np.ndarray): The indeces of the calibrators.
            calibrator_distance_modulus (np.ndarray): The distance modulus of the calibrators.

        Returns:
            np.ndarray: The residuals for the Tripp calibration.
        """
        
        # Calculate the absolute magnitude
        M_B = self.M_int + self.alpha * stretch + self.beta * color

        # Calculate the distance modulus
        mu = self.cosmo.distmod(redshift).value
        mu[calibrator_indeces] = calibrator_distance_modulus

        # Calculate the residuals
        residuals = apparent_B_mag - M_B - mu

        return residuals
    
    def covariance_matrix(
        self,
        redshifts: np.ndarray,
        observed_covariance: np.ndarray,
        calibratior_indeces: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the covariance matrix for the Tripp calibration.

        Args:
            redshifts (np.ndarray): The redshifts of the SNe.
            observed_covariance (np.ndarray): The observed covariance matrix.
            calibratior_indeces (np.ndarray): The indeces of the calibrators.

        Returns:
            np.ndarray: The covariance matrix for the Tripp calibration.
        """


        cov_dims = observed_covariance.shape
        n_cov_dims = len(cov_dims)
        n_sne = len(redshifts)

        if n_cov_dims == 1:
            observed_covariance = np.diag(observed_covariance)
        
        if n_cov_dims != 3:
            observed_covariance = np.expand_dims(observed_covariance, axis=0)
        
        if n_sne != cov_dims[0]:
            observed_covariance = np.tile(observed_covariance, (n_sne, 1, 1))

        cov = observed_covariance.copy()

        distmod_var = (
            (5 / np.log(10)) *
            (self.peculiar_velocity_dispersion / (SPEED_OF_LIGHT * redshifts))
        ) ** 2
        distmod_var[calibratior_indeces] = 0.0

        cov[:, 0, 0] += (
            self.sigma_M_int ** 2 +
            self.alpha ** 2 * self.sigma_stretch_int ** 2 +
            self.beta ** 2 * self.sigma_color_int ** 2 +
            distmod_var
        )
        cov[:, 1, 1] += self.sigma_stretch_int ** 2
        cov[:, 2, 2] += self.sigma_color_int ** 2
        cov[:, 0, 1] += self.alpha * self.sigma_stretch_int ** 2
        cov[:, 0, 2] += self.beta * self.sigma_color_int ** 2
        cov[:, 1,0] = cov[:, 0, 1]
        cov[:, 2, 0] = cov[:, 0, 2]

        return cov
        

class TrippDustCalibration():

    def __init__():
        pass