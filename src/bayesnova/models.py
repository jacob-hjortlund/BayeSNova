import inspect
import warnings
import numpy as np
import numba as nb
import bayesnova.utils as utils
import scipy.stats as stats
import scipy.special as special
import astropy.cosmology as cosmo

from functools import partial
from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags, ldqag
from typing import Callable

import bayesnova.preprocessing as prep
import bayesnova.cosmology_utils as cosmo_utils

NULL_VALUE = -9999.0
H0_CONVERSION_FACTOR = 0.001022
DH_70 = 4282.7494
SPEED_OF_LIGHT = 299792.458 # km/s
PECULIAR_VELOCITY_DISPERSION = 300 # km/s

# --------------------------------------- SUPERNOVA MODEL ---------------------------------------

# ---------- E(B-V) PRIOR INTEGRAL ------------

@nb.jit()
def EBV_integral_body(
    x, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, r1, r2, r3,
    selection_bias_correction,
    RB, sig_RB, tau_EBV, gamma_EBV,
) -> float:
    """
    Calculates the integrand of the E(B-V) prior.

    Args:
        x (float): E(B-V)/tau_EBV value.
        i1 (float): Covariance matrix element [0,0].
        i2 (float): Covariance matrix element [0,1].
        i3 (float): Covariance matrix element [0,2].
        i4 (float): Covariance matrix element [1,0].
        i5 (float): Covariance matrix element [1,1].
        i6 (float): Covariance matrix element [1,2].
        i7 (float): Covariance matrix element [2,0].
        i8 (float): Covariance matrix element [2,1].
        i9 (float): Covariance matrix element [2,2].
        r1 (float): Residual vector element [0].
        r2 (float): Residual vector element [1].
        r3 (float): Residual vector element [2].
        selection_bias_correction (float): Selection bias correction factor.
        RB (float): Mean extinction coefficient.
        sig_RB (float): Standard deviation of extinction coefficient.
        tau_EBV (float): E(B-V) scale factor.
        gamma_EBV (float): E(B-V) exponent.

    Returns:
        float: Value of the integrand.
    """

    # update res and cov
    r1 -= RB * tau_EBV * x
    r3 -= tau_EBV * x
    i1 += sig_RB * sig_RB * tau_EBV * tau_EBV * x * x
    i1 *= selection_bias_correction

    # precalcs
    exponent = gamma_EBV - 1
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
    r_inv_cov_r = 1./det * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 + 2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    value = np.exp(-0.5 * r_inv_cov_r - x + exponent * np.log(x) - 0.5 * logdet - 0.5 * np.log(2 * np.pi))

    return value

@nb.cfunc(quadpack_sig)
def EBV_integral(x, data):
    """
    Calculates the integrand of the E(B-V) prior.

    Args:
        x (float): E(B-V)/tau_EBV value.
        data (tuple): Tuple of data values.

    Returns:
        float: Value of the integrand.
    """

    _data = nb.carray(data, (17,))
    i1 = _data[0]
    i2 = _data[1]
    i3 = _data[2]
    i4 = _data[3]
    i5 = _data[4]
    i6 = _data[5]
    i7 = _data[6]
    i8 = _data[7]
    i9 = _data[8]
    r1 = _data[9]
    r2 = _data[10]
    r3 = _data[11]
    selection_bias_correction = _data[12]
    RB = _data[13]
    sig_RB = _data[14]
    tau_EBV = _data[15]
    gamma_EBV = _data[16]
    return EBV_integral_body(
        x, i1, i2, i3, i4, i5, i6, i7, i8, i9, r1, r2, r3,
        selection_bias_correction,
        RB, sig_RB, tau_EBV, gamma_EBV
    )
EBV_integral_ptr = EBV_integral.address

@nb.njit
def EBV_prior_marginalization(
    cov: np.ndarray, res: np.ndarray,
    RB: float, sig_RB: float, 
    tau_EBV: float, gamma_EBV: float, 
    lower_bound_EBV: float,
    upper_bound_EBV: float,
    selection_bias_correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the marginalization of the E(B-V) prior.

    Args:
        cov (np.ndarray): SNe covariance matrix.
        res (np.ndarray): SNe residual vector.
        RB (float): Mean extinction coefficient.
        sig_RB (float): Standard deviation of extinction coefficient.
        tau_EBV (float): E(B-V) scale factor.
        gamma_EBV (float): E(B-V) exponent.
        lower_bound_EBV (float): Lower bound of E(B-V) integral.
        upper_bound_EBV (float): Upper bound of E(B-V) integral.
        selection_bias_correction (np.ndarray): Selection bias correction.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of marginalization values and integration statuses.
    """

    n_sn = len(cov)
    params = np.array([
        RB, sig_RB, tau_EBV, gamma_EBV
    ])
        
    for i in range(n_sn):

        bias_corr = np.array([selection_bias_correction[i]])
        tmp_params = np.concatenate((
            cov[i].ravel(), res[i].ravel(),
            bias_corr, params
        )).copy()
        tmp_params.astype(np.float64)
        
        probs, _, status, _ = dqags(
            funcptr=EBV_integral_ptr, a=lower_bound_EBV,
            b=upper_bound_EBV, data=tmp_params
        )

    return probs, status

# ---------- E(B-V) PRIOR LOG-SPACE INTEGRAL ------------

@nb.jit()
def EBV_integral_log_body(
    x, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, r1, r2, r3,
    selection_bias_correction,
    RB, sig_RB, tau_EBV, gamma_EBV,
):
    """
    Calculates the log integrand of the E(B-V) prior.

    Args:
        x (float): E(B-V)/tau_EBV value.
        i1 (float): Covariance matrix element [0,0].
        i2 (float): Covariance matrix element [0,1].
        i3 (float): Covariance matrix element [0,2].
        i4 (float): Covariance matrix element [1,0].
        i5 (float): Covariance matrix element [1,1].
        i6 (float): Covariance matrix element [1,2].
        i7 (float): Covariance matrix element [2,0].
        i8 (float): Covariance matrix element [2,1].
        i9 (float): Covariance matrix element [2,2].
        r1 (float): Residual vector element [0].
        r2 (float): Residual vector element [1].
        r3 (float): Residual vector element [2].
        selection_bias_correction (float): Selection bias correction factor.
        RB (float): Mean extinction coefficient.
        sig_RB (float): Standard deviation of extinction coefficient.
        tau_EBV (float): E(B-V) scale factor.
        gamma_EBV (float): E(B-V) exponent.

    Returns:
        float: Value of the integrand.
    """

    # update res and cov
    r1 -= RB * tau_EBV * x
    r3 -= tau_EBV * x
    i1 += sig_RB * sig_RB * tau_EBV * tau_EBV * x * x
    i1 *= selection_bias_correction

    # precalcs
    exponent = gamma_EBV - 1
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
    r_inv_cov_r = 1./det * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 + 2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    value = -0.5 * r_inv_cov_r - x + exponent * np.log(x) - 0.5 * logdet - 0.5 * np.log(2 * np.pi)

    return value

@nb.cfunc(quadpack_sig)
def EBV_log_integral(x, data):
    """
    Calculates the log integrand of the E(B-V) prior.

    Args:
        x (float): E(B-V)/tau_EBV value.
        data (tuple): Tuple of data values.

    Returns:
        float: Value of the integrand.
    """

    _data = nb.carray(data, (17,))
    i1 = _data[0]
    i2 = _data[1]
    i3 = _data[2]
    i4 = _data[3]
    i5 = _data[4]
    i6 = _data[5]
    i7 = _data[6]
    i8 = _data[7]
    i9 = _data[8]
    r1 = _data[9]
    r2 = _data[10]
    r3 = _data[11]
    selection_bias_correction = _data[12]
    RB = _data[13]
    sig_RB = _data[14]
    tau_EBV = _data[15]
    gamma_EBV = _data[16]
    return EBV_integral_log_body(
        x, i1, i2, i3, i4, i5, i6, i7, i8, i9, r1, r2, r3,
        selection_bias_correction,
        RB, sig_RB, tau_EBV, gamma_EBV
    )
EBV_log_integral_ptr = EBV_log_integral.address

@nb.njit
def EBV_prior_log_marginalization(
    cov: np.ndarray, res: np.ndarray,
    RB: float, sig_RB: float,
    tau_EBV: float, gamma_EBV: float, 
    lower_bound_EBV: float,
    upper_bound_EBV: float,
    selection_bias_correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the log marginalization of the E(B-V) prior.

    Args:
        cov (np.ndarray): SNe covariance matrices.
        res (np.ndarray): SNe residual vectors.
        RB (float): Mean extinction coefficient.
        sig_RB (float): Standard deviation of extinction coefficient.
        tau_EBV (float): E(B-V) scale factor.
        gamma_EBV (float): E(B-V) exponent.
        lower_bound_EBV (float): Lower bound of E(B-V) prior.
        upper_bound_EBV (float): Upper bound of E(B-V) prior.
        selection_bias_correction (np.ndarray): Selection bias correction.

    Returns:
        tuple[np.ndarray, np.ndarray]: Log marginalization and status.
    """

    n_sn = len(cov)
    params = np.array([
        RB, sig_RB, tau_EBV, gamma_EBV
    ])
        
    for i in range(n_sn):

        bias_corr = np.array([selection_bias_correction[i]])
        tmp_params = np.concatenate((
            cov[i].ravel(), res[i].ravel(),
            bias_corr, params
        )).copy()
        tmp_params.astype(np.float64)
        
        logprobs, _, status, _ = ldqag(
            funcptr=EBV_log_integral_ptr, a=lower_bound_EBV,
            b=upper_bound_EBV, data=tmp_params
        )

    return logprobs, status

def sn_covariance(
    observational_covariances: np.ndarray,
    sn_redshifts: np.ndarray,
    calibrator_indices: np.ndarray,
    alpha: float, beta: float, sig_int: float,
    sig_x1: float, sig_c: float
) -> np.ndarray:
    """Given the observational covariances, redshifts, and calibrator indices,
    calculate the SN model covariance matrix for a given set of parameters.

    Args:
        observational_covariances (np.ndarray): Observational covariance matrix
        sn_redshifts (np.ndarray): Observed redshifts
        calibrator_indices (np.ndarray): Boolean array indicating which SNe are calibrators
        alpha (float): Tripp alpha parameter
        beta (float): Tripp beta parameter
        sig_int (float): Intrinsic scatter
        sig_x1 (float): Population scatter in stretch
        sig_c (float): Population scatter in intrinsic color

    Returns:
        np.ndarray: SN model covariance matrix
    """

    z = sn_redshifts.copy
    cov = observational_covariances.copy()

    cov[0,0] += (
        sig_int**2 + alpha**2 * sig_x1**2 + beta**2 * sig_c**2
    )

    distmod_var = z**2 * (
        (5. / np.log(10.))**2 * (PECULIAR_VELOCITY_DISPERSION / SPEED_OF_LIGHT)**2
    )
    distmod_var[calibrator_indices] = 0.
    cov[0,0] += distmod_var

    cov[1,1] += sig_x1**2
    cov[2,2] += sig_c**2
    cov[0,1] += alpha * sig_x1**2
    cov[0,2] += beta * sig_c**2
    cov[1,0] = cov[0,1]
    cov[2,0] = cov[0,2]

    return cov

def sn_residuals(
    sn_app_magnitudes: np.ndarray,
    sn_stretch: np.ndarray,
    sn_colors: np.ndarray,
    sn_redshifts: np.ndarray,
    MB: float, x1: float, c_int: float,
    alpha: float, beta: float,
    cosmology: cosmo.Cosmology,
    calibrator_indices: np.ndarray = None,
    calibrator_distance_moduli: np.ndarray = None
) -> np.ndarray:
    """Given the observed SN magnitudes, stretch, color, redshift, and
    parameters, calculate the SN model residuals.

    Args:
        sn_app_magnitudes (np.ndarray): Observed SN apparent B-band magnitudes
        sn_stretch (np.ndarray): Observed SN stretch
        sn_colors (np.ndarray): Observed SN apparent color
        sn_redshifts (np.ndarray): Observed SN redshifts
        MB (float): Absolute B-band magnitude
        x1 (float): Population stretch
        c_int (float): Population intrinsic color
        alpha (float): Tripp alpha parameter
        beta (float): Tripp beta parameter
        cosmology (cosmo.Cosmology): Cosmology object
        calibrator_indices (np.ndarray, optional): Boolean array indicating which SNe are calibrators. Defaults to None.
        calibrator_distance_moduli (np.ndarray, optional): Distance moduli of calibrators. Defaults to None.

    Returns:
        np.ndarray: SN model residuals
    """
    
    n_sn = len(sn_app_magnitudes)
    residuals = np.zeros((n_sn, 3))

    no_calibrator_indices = calibrator_indices is None
    no_calibrator_distance_moduli = calibrator_distance_moduli is None
    if no_calibrator_indices and no_calibrator_distance_moduli:
        calibrator_indices = np.zeros(n_sn, dtype='bool')
        calibrator_distance_moduli = 0.
    if (
        (no_calibrator_indices and not no_calibrator_distance_moduli) or
        (not no_calibrator_indices and no_calibrator_distance_moduli)
    ):
        raise ValueError(
            "Must provide both calibrator indices and calibrator distance moduli"
        )

    distance_moduli = cosmology.distmod(sn_redshifts).value
    distance_moduli[calibrator_indices] = calibrator_distance_moduli

    residuals[:,0] = sn_app_magnitudes - (
        MB + alpha * x1 + beta * c_int + distance_moduli
    )
    residuals[:,1] = sn_stretch - x1
    residuals[:,2] = sn_colors - c_int

    return residuals

def marginalize_EBV(
    EBV_marginalization_func: Callable,
    sn_covariances: np.ndarray,
    sn_residuals: np.ndarray,
    RB: float, sig_RB: float,
    tau_EBV: float, gamma_EBV: float,
    lower_bound_EBV: float,
    upper_bound_EBV: float,
    selection_bias_correction: np.ndarray
) -> tuple:
    """Given the SN covariance and residuals, calculate the log likelihood
    marginalized over EBV.

    Args:
        EBV_marginalization_func (Callable): Function to calculate the log likelihood marginalized over EBV
        sn_covariances (np.ndarray): SN covariance matrix
        sn_residuals (np.ndarray): SN residuals
        RB (float): Mean extinction coefficient
        sig_RB (float): Scatter in extinction coefficient
        tau_EBV (float): EBV scale factor
        gamma_EBV (float): EBV exponent
        lower_bound_EBV (float): Lower bound on EBV
        upper_bound_EBV (float): Upper bound on EBV
        selection_bias_correction (np.ndarray): Selection bias correction

    Returns:
        tuple[np.ndarray, np.ndarray]: Log likelihood marginalized over EBV, status array
    """
    
    log_truncated_normalization = (
        np.log(special.gammainc(gamma_EBV, upper_bound_EBV)) +
        special.loggamma(gamma_EBV)
    )

    logprobs, status = EBV_marginalization_func(
        cov=sn_covariances, res=sn_residuals,
        RB=RB, sig_RB=sig_RB, tau_EBV=tau_EBV,
        gamma_EBV=gamma_EBV, lower_bound_EBV=lower_bound_EBV,
        upper_bound_EBV=upper_bound_EBV,
        selection_bias_correction=selection_bias_correction
    )

    logprobs -= log_truncated_normalization

    return logprobs, status