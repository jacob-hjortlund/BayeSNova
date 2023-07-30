import numpy as np
import numba as nb
import scipy.stats as stats
import scipy.special as special
import astropy.cosmology as cosmo

from typing import Union
from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags, ldqag
from base_models import Gaussian, Mixture, Weighting, ConstantWeighting

NULL_VALUE = -9999.0
H0_CONVERSION_FACTOR = 0.001022
DH_70 = 4282.7494
SPEED_OF_LIGHT = 299792.458 # km/s

# ---------- E(B-V)_i MARGINALIZATION INTEGRAL ------------

@nb.jit()
def _E_BV_i_log_integral_body(
    x, cov_i1, cov_i2, cov_i3, cov_i4,
    cov_i5, cov_i6, cov_i7, cov_i8, cov_i9,
    res_i1, res_i2, res_i3,
    selection_bias_correction,
    R_B, sigma_R_B, tau_Ebv, gamma_Ebv,
):  

    # update res and cov
    res_i1 -= R_B * tau_Ebv * x
    res_i3 -= tau_Ebv * x
    cov_i1 += sigma_R_B * sigma_R_B * tau_Ebv * tau_Ebv * x * x
    cov_i1 *= selection_bias_correction

    # precalcs
    exponent = gamma_Ebv - 1
    A1 = cov_i5 * cov_i9 - cov_i6 * cov_i6
    A2 = cov_i6 * cov_i3 - cov_i2 * cov_i9
    A3 = cov_i2 * cov_i6 - cov_i5 * cov_i3
    A5 = cov_i1 * cov_i9 - cov_i3 * cov_i3
    A6 = cov_i2 * cov_i3 - cov_i1 * cov_i6
    A9 = cov_i1 * cov_i5 - cov_i2 * cov_i2
    det = cov_i1 * A1 + cov_i2 * A2 + cov_i3 * A3

    if det < 0:
        cov = np.array([
            [cov_i1, cov_i2, cov_i3],
            [cov_i4, cov_i5, cov_i6],
            [cov_i7, cov_i8, cov_i9]
        ])
        eigvals = np.linalg.eigvalsh(cov)
        cov += np.eye(3) * np.abs(np.min(eigvals)) * (1 + 1e-2)
        cov_i1, cov_i2, cov_i3, cov_i4, cov_i5, cov_i6, cov_i7, cov_i8, cov_i9 = cov.flatten()
        A1 = cov_i5 * cov_i9 - cov_i6 * cov_i6
        A2 = cov_i6 * cov_i3 - cov_i2 * cov_i9
        A3 = cov_i2 * cov_i6 - cov_i5 * cov_i3
        A5 = cov_i1 * cov_i9 - cov_i3 * cov_i3
        A6 = cov_i2 * cov_i3 - cov_i1 * cov_i6
        A9 = cov_i1 * cov_i5 - cov_i2 * cov_i2
        det = cov_i1 * A1 + cov_i2 * A2 + cov_i3 * A3
    
    logdet = np.log(det)

    # # calculate prob
    r_inv_cov_r = (
        1./det * (res_i1 * res_i1 * A1 + res_i2 * res_i2 * A5 + res_i3 * res_i3 * A9 +
                  2 * (res_i1 * res_i2 * A2 + res_i1 * res_i3 * A3 + res_i2 * res_i3 * A6))
    )
    value = -0.5 * r_inv_cov_r - x + exponent * np.log(x) - 0.5 * logdet - 0.5 * np.log(2 * np.pi)

    return value

@nb.cfunc(quadpack_sig)
def _E_BV_i_log_integral(x, data):
    _data = nb.carray(data, (17,))
    cov_i1 = _data[0]
    cov_i2 = _data[1]
    cov_i3 = _data[2]
    cov_i4 = _data[3]
    cov_i5 = _data[4]
    cov_i6 = _data[5]
    cov_i7 = _data[6]
    cov_i8 = _data[7]
    cov_i9 = _data[8]
    res_i1 = _data[9]
    res_i2 = _data[10]
    res_i3 = _data[11]
    selection_bias_correction = _data[12]
    R_B = _data[13]
    sigma_R_B = _data[14]
    tau_E_BV = _data[15]
    gamma_E_BV = _data[16]
    return _E_BV_i_log_integral_body(
        x, cov_i1, cov_i2, cov_i3, cov_i4,
        cov_i5, cov_i6, cov_i7, cov_i8, cov_i9, 
        res_i1, res_i2, res_i3,
        selection_bias_correction,
        R_B, sigma_R_B, tau_E_BV, gamma_E_BV,
    )
_E_BV_i_log_integral_ptr = _E_BV_i_log_integral.address

@nb.jit()
def _E_BV_i_integral_body(
    x, cov_i1, cov_i2, cov_i3, cov_i4,
    cov_i5, cov_i6, cov_i7, cov_i8, cov_i9,
    res_i1, res_i2, res_i3,
    selection_bias_correction,
    R_B, sigma_R_B, tau_E_BV, gamma_E_BV,
):
    
    log_integral = _E_BV_i_log_integral_body(
        x, cov_i1, cov_i2, cov_i3, cov_i4,
        cov_i5, cov_i6, cov_i7, cov_i8, cov_i9, 
        res_i1, res_i2, res_i3,
        selection_bias_correction,
        R_B, sigma_R_B, tau_E_BV, gamma_E_BV,
    )

    return np.exp(log_integral)

@nb.cfunc(quadpack_sig)
def _E_BV_i_integral(x, data):
    _data = nb.carray(data, (17,))
    cov_i1 = _data[0]
    cov_i2 = _data[1]
    cov_i3 = _data[2]
    cov_i4 = _data[3]
    cov_i5 = _data[4]
    cov_i6 = _data[5]
    cov_i7 = _data[6]
    cov_i8 = _data[7]
    cov_i9 = _data[8]
    res_i1 = _data[9]
    res_i2 = _data[10]
    res_i3 = _data[11]
    selection_bias_correction = _data[12]
    R_B = _data[13]
    sigma_R_B = _data[14]
    tau_E_BV = _data[15]
    gamma_E_BV = _data[16]
    return _E_BV_i_integral_body(
        x, cov_i1, cov_i2, cov_i3, cov_i4,
        cov_i5, cov_i6, cov_i7, cov_i8, cov_i9,
        res_i1, res_i2, res_i3,
        selection_bias_correction,
        R_B, sigma_R_B, tau_E_BV, gamma_E_BV,
    )
_E_BV_i_integral_ptr = _E_BV_i_integral.address

#@nb.njit
def _E_BV_marginalization(
    covariance: np.ndarray, residual: np.ndarray,
    R_B: float, sigma_R_B: float, 
    tau_E_BV: float, gamma_E_BV: float,
    upper_bound_E_BV: float,
    selection_bias_correction: np.ndarray,
):
    """
    Calculate the marginalization integral for the E(B-V) parameter.

    Args:
        covariance (np.ndarray): The covariance matrix for the SNe.
        residual (np.ndarray): The residual for the SNe.
        R_B (float): The R_B parameter.
        sigma_R_B (float): The sigma_R_B parameter.
        tau_E_BV (float): The tau_E_BV parameter.
        gamma_E_BV (float): The gamma_E_BV parameter.
        upper_bound_E_BV (float): The upper bound for the E(B-V) integral.
        selection_bias_correction (np.ndarray): The selection bias correction.

    Returns:
        np.ndarray: The marginalization integral for the E(B-V) parameter.
    """

    n_sne = len(covariance)
    probs = np.zeros(n_sne)
    status = np.ones(n_sne, dtype='bool')
    params = np.array([
        R_B, sigma_R_B, tau_E_BV, gamma_E_BV
    ])
        
    for i in range(n_sne):
        bias_corr = [selection_bias_correction[i]]
        inputs_i = np.concatenate((
            covariance[i].ravel(),
            residual[i].ravel(),
            bias_corr,
            params
        )).copy()
        inputs_i.astype(np.float64)
        
        prob_i, _, status_i, _ = dqags(
            funcptr=_E_BV_i_integral_ptr, a=0,
            b=upper_bound_E_BV, data=inputs_i
        )


        probs[i] = prob_i
        status[i] = status_i

    return probs, status

@nb.njit
def _E_BV_log_marginalization(
    covariance: np.ndarray, residual: np.ndarray,
    R_B: float, sigma_R_B: float,
    tau_E_BV: float, gamma_E_BV: float,
    upper_bound_E_BV: float,
    selection_bias_correction: np.ndarray,
):
    """
    Calculate the log likelihood using log marginalization of the latent E(B-V) parameter.

    Args:
        covariance (np.ndarray): The covariance matrix for the SNe.
        residual (np.ndarray): The residual for the SNe.
        R_B (float): The R_B parameter.
        sigma_R_B (float): The sigma_R_B parameter.
        tau_E_BV (float): The tau_E_BV parameter.
        gamma_E_BV (float): The gamma_E_BV parameter.
        upper_bound_E_BV (float): The upper bound for the E(B-V) integral.
        selection_bias_correction (np.ndarray): The selection bias correction.

    Returns:
        np.ndarray: The log marginalization integral for the E(B-V) parameter.
    """
    
    n_sne = len(covariance)
    log_probs = np.zeros(n_sne)
    status = np.ones(n_sne, dtype='bool')
    params = np.array([
        R_B, sigma_R_B, tau_E_BV, gamma_E_BV
    ])

    for i in range(n_sne):
        bias_corr = [selection_bias_correction[i]]
        inputs_i = np.concatenate((
            covariance[i].ravel(),
            residual[i].ravel(),
            bias_corr,
            params
        )).copy()
        inputs_i.astype(np.float64)
        
        log_prob_i, _, status_i, _ = ldqag(
            funcptr=_E_BV_i_log_integral_ptr, a=0,
            b=upper_bound_E_BV, data=inputs_i
        )

        log_probs[i] = log_prob_i
        status[i] = status_i

    return log_probs, status


# ---------------------- MODELS ----------------------------

class OldTripp(Gaussian):

    def __init__(
        self,
        M_int: float = -19.3,
        sigma_M_int: float = 0.1,
        alpha: float = 0.141,
        beta: float = 3.101,
        **kwargs,
    ):
        
        self.M_int = M_int
        self.sigma_M_int = sigma_M_int
        self.alpha = alpha
        self.beta = beta
        self.peculiar_velocity_dispersion = kwargs.get(
            'peculiar_velocity_dispersion',
            250.0
        )
        self.cosmo = kwargs.get(
            'cosmology',
            None
        )
        if self.cosmo is None:
            raise ValueError('A Cosmology model must be provided.')
    
    def residual(
        self,
        apparent_B_mag: np.ndarray,
        stretch: np.ndarray,
        color: np.ndarray,
        redshift: np.ndarray,
        calibrator_indeces: np.ndarray,
        calibrator_distance_modulus: np.ndarray,
    ):
        
        mu = self.cosmo.distance_modulus(redshift)
        residuals = (
            apparent_B_mag -
            self.M_int -
            self.alpha * stretch -
            self.beta * color -
            mu
        )

        return residuals
    
    def covariance(
        self,
        redshift: np.ndarray,
        observed_covariance: np.ndarray,
        calibratior_indeces: np.ndarray,
    ):
        
        cov_tmp = np.diagonal(
            observed_covariance,
            axis1=1, axis2=2
        )
        cov = cov_tmp.copy()
        cov[:,1] = cov_tmp[:,1] * self.alpha**2
        cov[:,2] = cov_tmp[:,2] * self.beta**2
        cov = np.sum(cov, axis=1)
        cov += (
            (5 / np.log(10)) *
            (self.peculiar_velocity_dispersion / (SPEED_OF_LIGHT * redshift))
        )**2
        cov -= 2 * self.beta * observed_covariance[:, 0, 2]
        cov += 2 * self.alpha * observed_covariance[:, 0, 1]
        cov -= 2 * self.alpha * self.beta * observed_covariance[:, 1, 2]
        cov += self.sigma_M_int ** 2

        return cov

    def log_likelihood(
        self,
        apparent_B_mag: np.ndarray,
        stretch: np.ndarray,
        color: np.ndarray,
        redshift: np.ndarray,
        observed_covariance: np.ndarray,
        calibrator_indeces: np.ndarray,
        calibrator_distance_modulus: np.ndarray,
    ):

        residual = self.residual(
            apparent_B_mag=apparent_B_mag,
            stretch=stretch,
            color=color,
            redshift=redshift,
            calibrator_indeces=calibrator_indeces,
            calibrator_distance_modulus=calibrator_distance_modulus,
        )
    
        covariance = self.covariance(
            redshift=redshift,
            observed_covariance=observed_covariance,
            calibratior_indeces=calibrator_indeces,
        )


        log_likelihood = -0.5 * (residual**2 / covariance + np.log(covariance))
    
        return log_likelihood

class Tripp(Gaussian):

    def __init__(
        self,
        M_int: float = -19.3,
        sigma_M_int: float = 0.1,
        alpha: float = 0.141,
        stretch_int: float = 1.0,
        sigma_stretch_int: float = 0.0,
        beta: float = 3.101,
        color_int: float = -0.1,
        sigma_color_int: float = 0.0,
        peculiar_velocity_dispersion: float = 200.0,
        **kwargs,
    ):
        
        self.M_int = M_int
        self.sigma_M_int = sigma_M_int
        self.alpha = alpha
        self.stretch_int = stretch_int
        self.sigma_stretch_int = sigma_stretch_int
        self.beta = beta
        self.color_int = color_int
        self.sigma_color_int = sigma_color_int
        self.peculiar_velocity_dispersion = peculiar_velocity_dispersion
        self.cosmo = kwargs.get(
            'cosmology',
            None
        )
        if self.cosmo is None:
            raise ValueError('A Cosmology model must be provided.')
    
    def residual(
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
        M_B = self.M_int + self.alpha * self.stretch_int + self.beta * self.color_int

        # Calculate the distance modulus
        mu = self.cosmo.distance_modulus(redshift)
        mu[calibrator_indeces] = calibrator_distance_modulus

        # Calculate the residuals
        m_B_residual = apparent_B_mag - M_B - mu
        stretch_residual = stretch - self.stretch_int
        color_residual = color - self.color_int

        residuals = np.hstack(
            [m_B_residual[:, None], stretch_residual[:, None], color_residual[:, None]]
        )
        residuals = np.atleast_3d(residuals)

        return residuals
    
    def covariance(
        self,
        redshift: np.ndarray,
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
        n_sne = len(redshift)

        if n_cov_dims == 1:
            observed_covariance = np.diag(observed_covariance)
        
        if n_cov_dims != 3:
            observed_covariance = np.expand_dims(observed_covariance, axis=0)
        
        if n_sne != cov_dims[0]:
            observed_covariance = np.tile(observed_covariance, (n_sne, 1, 1))

        cov = observed_covariance.copy()

        distmod_var = (
            (5 / np.log(10)) *
            (self.peculiar_velocity_dispersion / (SPEED_OF_LIGHT * redshift))
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
        cov[:, 1, 0] = cov[:, 0, 1]
        cov[:, 2, 0] = cov[:, 0, 2]

        return cov
    
    def log_likelihood(
        self,
        apparent_B_mag: np.ndarray,
        stretch: np.ndarray,
        color: np.ndarray,
        redshift: np.ndarray,
        observed_covariance: np.ndarray,
        calibrator_indeces: np.ndarray,
        calibrator_distance_modulus: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the log likelihood for the Tripp calibration.

        Args:
            apparent_B_mag (np.ndarray): The apparent B-band magnitudes.
            stretch (np.ndarray): The stretch of the SNe.
            color (np.ndarray): The color of the SNe.
            redshift (np.ndarray): The redshift of the SNe.
            observed_covariance (np.ndarray): The observed covariance matrix.
            calibrator_indeces (np.ndarray): The indeces of the calibrators.
            calibrator_distance_modulus (np.ndarray): The distance modulus of the calibrators.

        Returns:
            np.ndarray: The log likehood for the Tripp calibration with shape (n_sne,).
        """
        
        residual = self.residual(
            apparent_B_mag=apparent_B_mag,
            stretch=stretch,
            color=color,
            redshift=redshift,
            calibrator_indeces=calibrator_indeces,
            calibrator_distance_modulus=calibrator_distance_modulus,
        )

        covariance = self.covariance(
            redshift=redshift,
            observed_covariance=observed_covariance,
            calibratior_indeces=calibrator_indeces,
        )

        _, logdets = np.linalg.slogdet(covariance)
        exponent = np.squeeze(
            residual.transpose(0, 2, 1) @
            np.linalg.solve(covariance, residual)
        )

        log_likelihood = -.5 * (logdets + exponent + np.log(2 * np.pi))

        return log_likelihood
        
class TrippDust(Tripp):

    def __init__(
        self,
        M_int: float = -19.3,
        sigma_M_int: float = 0.1,
        alpha: float = 0.141,
        stretch_int: float = 1.0,
        sigma_stretch_int: float = 0.0,
        beta: float = 3.101,
        color_int: float = -0.1,
        sigma_color_int: float = 0.0,
        R_B: float = 3.1,
        sigma_R_B: float = 0.0,
        gamma_E_BV: float = 1.,
        tau_E_BV: float = 1.,
        peculiar_velocity_dispersion: float = 200.,
        **kwargs,
    ):
        
        super().__init__(
            M_int=M_int,
            sigma_M_int=sigma_M_int,
            alpha=alpha,
            stretch_int=stretch_int,
            sigma_stretch_int=sigma_stretch_int,
            beta=beta,
            color_int=color_int,
            sigma_color_int=sigma_color_int,
            peculiar_velocity_dispersion=peculiar_velocity_dispersion,
            **kwargs,
        )

        self.R_B = R_B
        self.sigma_R_B = sigma_R_B
        self.gamma_E_BV = gamma_E_BV
        self.tau_E_BV = tau_E_BV

    def get_upper_bound_E_BV(
        self,
        upper_bound_E_BV: Union[float, np.ndarray],
    ) -> float:
        """
        Get the upper bound for the E(B-V) integral. If the upper bound is a (2, N) array of 
        gamma values and upper bounds, the upper bound for the E(B-V) integral is the upper bound
        corresponding to the gamma value closest to the gamma value of the model. If the upper bound
        is a float, it is returned as is.

        Args:
            upper_bound_E_BV (Union[float, np.ndarray]): The upper bound for the E(B-V) integral.

        Returns:
            float: The upper bound for the E(B-V) integral.
        """
        
        if isinstance(upper_bound_E_BV, float):
            return upper_bound_E_BV
        
        gamma_values = upper_bound_E_BV[0]
        bounds = upper_bound_E_BV[1]
        idx_nearest = np.argmin(np.abs(gamma_values - self.gamma_E_BV))
        upper_bound = bounds[idx_nearest]

        return upper_bound

    def log_likelihood(
        self,
        apparent_B_mag: np.ndarray,
        stretch: np.ndarray,
        color: np.ndarray,
        redshift: np.ndarray,
        observed_covariance: np.ndarray,
        calibrator_indeces: np.ndarray,
        calibrator_distance_modulus: np.ndarray,
        selection_bias_correction: np.ndarray = None,
        upper_bound_E_BV: Union[float, np.ndarray] = 10.0,
        use_log_marginalization: bool = False,
    ) -> np.ndarray:
        """
        Calculate the log likehood for the Tripp calibration with dust.

        Args:
            apparent_B_mag (np.ndarray): The apparent B-band magnitudes.
            stretch (np.ndarray): The stretch of the SNe.
            color (np.ndarray): The color of the SNe.
            redshift (np.ndarray): The redshift of the SNe.
            observed_covariance (np.ndarray): The observed covariance matrix.
            calibrator_indeces (np.ndarray): The indeces of the calibrators.
            calibrator_distance_modulus (np.ndarray): The distance modulus of the calibrators.
            selection_bias_correction (np.ndarray, optional): The selection bias correction. Defaults to None.
            upper_bound_E_BV (Union[float, np.ndarray], optional): The upper bound for the E(B-V) integral. Is
                a float or a (2, N) array of gamma values and upper bounds. Defaults to 10.0.
            use_log_marginalization (bool, optional): Whether to use log marginalization. Defaults to False.

        Returns:
            np.ndarray: The log likehood for the Tripp calibration with dust with shape (n_sne,).
        """

        residual = self.residual(
            apparent_B_mag=apparent_B_mag,
            stretch=stretch,
            color=color,
            redshift=redshift,
            calibrator_indeces=calibrator_indeces,
            calibrator_distance_modulus=calibrator_distance_modulus,
        )

        covariance = self.covariance(
            redshift=redshift,
            observed_covariance=observed_covariance,
            calibratior_indeces=calibrator_indeces,
        )

        marginalization_func = _E_BV_log_marginalization if use_log_marginalization else _E_BV_marginalization
        
        upper_bound_E_BV = self.get_upper_bound_E_BV(
            upper_bound_E_BV=upper_bound_E_BV,
        )

        E_BV_norm = (
            np.log(special.gammainc(self.gamma_E_BV, upper_bound_E_BV)) +
            special.loggamma(self.gamma_E_BV)
        )

        if selection_bias_correction is None:
            selection_bias_correction = np.ones_like(redshift)

        # TODO: Add debug logging for marginalization status
        E_BV_marginalization, status = marginalization_func(
            covariance=covariance,
            residual=residual,
            R_B=self.R_B,
            sigma_R_B=self.sigma_R_B,
            tau_E_BV=self.tau_E_BV,
            gamma_E_BV=self.gamma_E_BV,
            upper_bound_E_BV=upper_bound_E_BV,
            selection_bias_correction=selection_bias_correction,
        )

        log_likehood = E_BV_marginalization
        if not use_log_marginalization:
            log_likehood = np.log(E_BV_marginalization)
        
        log_likehood -= E_BV_norm

        return log_likehood

class TwoSNPopulation(Mixture):

    def __init__(
        self,
        population_models: list[Gaussian],
        weighting_model: Gaussian,
    ):
        super().__init__(
            population_models=population_models,
            weighting_model=weighting_model,
        )
        self.n_populations = len(population_models)
        if self.n_populations != 2:
            raise ValueError(f"Expected 2 population models, got {self.n_populations}.")
    
    def log_likelihood(
        self,
        apparent_B_mag: np.ndarray,
        stretch: np.ndarray,
        color: np.ndarray,
        redshift: np.ndarray,
        observed_covariance: np.ndarray,
        calibrator_indeces: np.ndarray,
        calibrator_distance_modulus: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        
        log_likelihoods = np.zeros(
            shape=(len(apparent_B_mag), self.n_populations)
        )

        for i, population_model in enumerate(self.population_models):
            log_likelihoods[:, i] = population_model.log_likelihood(
                apparent_B_mag=apparent_B_mag,
                stretch=stretch,
                color=color,
                redshift=redshift,
                observed_covariance=observed_covariance,
                calibrator_indeces=calibrator_indeces,
                calibrator_distance_modulus=calibrator_distance_modulus,
                **kwargs,
            )
        
        population_1_weight = self.weighting_model.calculate_weight(
            redshift=redshift, **kwargs
        )
        population_2_weight = 1.0 - population_1_weight

        log_likelihoods = np.logaddexp(
            np.log(population_1_weight) + log_likelihoods[:, 0],
            np.log(population_2_weight) + log_likelihoods[:, 1],
        )

        return log_likelihoods