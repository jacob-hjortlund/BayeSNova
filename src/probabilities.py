import numpy as np
import scipy.special as sp_special
import scipy.integrate as sp_integrate
import src.utils as utils

from astropy.cosmology import Planck18_arXiv_v2 as cosmo

def _log_prior(
    mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, sig_int_1, rb_1, sig_rb_1, tau_1, alpha_g_1,
    mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, rb_2, sig_rb_2, tau_2, alpha_g_2
):

    return 1

def _population_covariance(
    alpha: float, beta: float, sig_s: float, sig_c: float, sig_int: float
) -> tuple:
    """Calculate the covariance matrix shared by all SN in a given population

    Args:
        alpha (float): Stretch correction
        beta (float): Intrinsic color correction
        sig_s (float): Stretch uncertainty
        sig_c (float): Intrinsic color uncertainty
        sig_int (float): Intrinsic scatter

    Returns:
        tuple: Shared covariance matrix
    """

    mu = np.zeros(3)
    cov = np.zeros((3,3))
    cov[0,0] = sig_int**2 + alpha**2 * sig_s**2 + beta**2 * sig_c**2
    cov[1,1] = sig_s**2
    cov[2,2] = sig_c**2
    cov[0,1] = alpha * sig_s**2
    cov[0,2] = beta * sig_c**2
    cov[1,0] = cov[0,1]
    cov[2,0] = cov[0,2]

    return cov

def _population_r(
    mb: np.ndarray, s: np.ndarray, c: np.ndarray, Mb: float, alpha_h: float, 
    beta_h: float, s_h: float, c_h: float, H0: float, z: np.ndarray
) -> np.ndarray:
    """Calculate residual between data and mean vector

    Args:
        mb (np.ndarray): Apparent magnitudes
        s (np.ndarray): SN stretches
        c (np.ndarray): SN intrinsic colors
        Mb (float): Population absolute magnitude
        alpha_h (float): Population stretch correction
        beta_h (float): Population intrinsic color correction
        s_h (float): Population stretch
        c_h (float): Population intrinsic color
        H0 (float): Hubble constant
        z (np.ndarray): SN redshifts

    Returns:
        np.ndarray: Residuals
    """

    r = np.zeros((len(mb), 3))
    distance_modulus = cosmo.distmod(z) + 5. * np.log10(cosmo.H0.value / H0)
    r[:, 0] = mb - (
        Mb + cosmo.distmod(z) + alpha_h * s + beta_h * c + distance_modulus
    ) 
    r[:, 1] = s - s_h
    r[:, 2] = c - c_h

    return r

def _log_likelihood(
    mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, rb_1, sig_rb_1, tau_1, alpha_g_1,
    mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, rb_2, sig_rb_2, tau_2, alpha_g_2
):

    return 1

def generate_log_prob(
    shared_par_names: list, independent_par_names: list, ratio_par_name: list,
    covariance: np.ndarray, z: np.ndarray
):
    def log_prob(theta):

        arg_dict = utils.theta_to_dict(
            theta, shared_par_names, independent_par_names, ratio_par_name 
        )

        log_prior = _log_prior(**arg_dict)
        if np.isinf(log_prior):
            return log_prior
        
        log_likelihood = _log_likelihood(**arg_dict)

        return log_likelihood + log_prior
    
    return log_prob