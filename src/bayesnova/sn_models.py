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

# ---------- E(B-V)_i MARGINALIZATION INTEGRAL ------------

@nb.jit()
def E_BV_i_integral_body(
    x, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, r1, r2, r3,
    selection_bias_correction,
    rb, sig_rb, tau_Ebv, gamma_Ebv,
):  

    # update res and cov
    r1 -= rb * tau_Ebv * x
    r3 -= tau_Ebv * x
    i1 += sig_rb * sig_rb * tau_Ebv * tau_Ebv * x * x
    i1 *= selection_bias_correction

    # precalcs
    exponent = gamma_Ebv - 1
    A1 = i5 * i9 - i6 * i6
    A2 = i6 * i3 - i2 * i9
    A3 = i2 * i6 - i5 * i3
    A5 = i1 * i9 - i3 * i3
    A6 = i2 * i3 - i1 * i6
    A9 = i1 * i5 - i2 * i2
    det = i1 * A1 + i2 * A2 + i3 * A3

    if det < 0:
        cov = np.array([
            [i1, i2, i3],
            [i4, i5, i6],
            [i7, i8, i9]
        ])
        eigvals = np.linalg.eigvalsh(cov)
        cov += np.eye(3) * np.abs(np.min(eigvals)) * (1 + 1e-2)
        i1, i2, i3, i4, i5, i6, i7, i8, i9 = cov.flatten()
        A1 = i5 * i9 - i6 * i6
        A2 = i6 * i3 - i2 * i9
        A3 = i2 * i6 - i5 * i3
        A5 = i1 * i9 - i3 * i3
        A6 = i2 * i3 - i1 * i6
        A9 = i1 * i5 - i2 * i2
        det = i1 * A1 + i2 * A2 + i3 * A3
    
    logdet = np.log(det)

    # # calculate prob
    r_inv_cov_r = (
        1./det * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 +
                  2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    )
    value = np.exp(
        -0.5 * r_inv_cov_r - x + exponent * np.log(x) - 0.5 * logdet - 0.5 * np.log(2 * np.pi)
    )

    return value

@nb.cfunc(quadpack_sig)
def E_BV_i_integral(x, data):
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
    return E_BV_i_integral_body(
        x, cov_i1, cov_i2, cov_i3, cov_i4,
        cov_i5, cov_i6, cov_i7, cov_i8, cov_i9,
        res_i1, res_i2, res_i3,
        selection_bias_correction,
        R_B, sigma_R_B, tau_E_BV, gamma_E_BV,
    )
E_BV_i_integral_ptr = E_BV_i_integral.address

@nb.njit
def _E_BV_marginalization(
    covariance: np.ndarray, residual: np.ndarray,
    R_B: float, sigma_R_B: float, 
    tau_E_BV: float, gamma_E_BV: float,
    upper_bound_E_BV: float,
    selection_bias_correction: np.ndarray,
):

    n_sne = len(covariance)
    logprobs = np.zeros(n_sne)
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
        
        logprob_i, _, status_i, _ = dqags(
            funcptr=E_BV_i_integral_ptr, a=0,
            b=upper_bound_E_BV, data=inputs_i
        )


        logprobs[i, 0] = logprob_i
        status[i, 0] = status_i

    return logprobs, status


# ---------------------- MODELS ----------------------------

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
        

class TrippDustCalibration(TrippCalibration):

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
        R_B: float = 3.1,
        sigma_R_B: float = 0.0,
        gamma_E_BV: float = 1.,
        tau_E_BV: float = 1.,
    ):
        
        super().__init__(
            H0=H0,
            Om0=Om0,
            w0=w0,
            wa=wa,
            M_int=M_int,
            sigma_M_int=sigma_M_int,
            alpha=alpha,
            stretch_int=stretch_int,
            sigma_stretch_int=sigma_stretch_int,
            beta=beta,
            color_int=color_int,
            sigma_color_int=sigma_color_int,
            peculiar_velocity_dispersion=peculiar_velocity_dispersion,
        )

        self.R_B = R_B
        self.sigma_R_B = sigma_R_B
        self.gamma_E_BV = gamma_E_BV
        self.tau_E_BV = tau_E_BV

