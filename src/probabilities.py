import numpy as np
import scipy.special as sp_special
import scipy.integrate as sp_integrate

import src.utils as utils

def _log_prior(
    mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, sig_int_1, rb_1, sig_rb_1, tau_1, alpha_g_1,
    mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, rb_2, sig_rb_2, tau_2, alpha_g_2
):

    return 1

def _population_covariance(
    alpha: float, beta: float, sig_s: float, sig_c: float, sig_int: float
) -> np.ndarray:
    """Calculate the covariance matrix shared by all SN in a given population

    Args:
        alpha (float): Stretch correction
        beta (float): Intrinsic color correction
        sig_s (float): Stretch uncertainty
        sig_c (float): Intrinsic color uncertainty
        sig_int (float): Intrinsic scatter

    Returns:
        np.ndarray: Shared covariance matrix
    """

    cov = np.zeros((3,3))
    cov[0,0] = sig_int**2 + alpha**2 * sig_s**2 + beta**2 * sig_c**2
    cov[1,1] = sig_s**2
    cov[2,2] = sig_c**2
    cov[0,1] = alpha * sig_s**2
    cov[0,2] = beta * sig_c**2
    cov[1,0] = cov[0,1]
    cov[2,0] = cov[0,2]

    return cov

def _log_likelihood(
    mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, rb_1, sig_rb_1, tau_1, alpha_g_1,
    mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, rb_2, sig_rb_2, tau_2, alpha_g_2
):

    return 1

def log_prob(theta, **kwargs):

    arg_dict = utils.theta_to_dict(
        theta=theta, **kwargs['model_par_cfg']
    )

    log_prior = _log_prior(**arg_dict)
    if np.isinf(log_prior):
        return log_prior
    
    log_likelihood = _log_likelihood(**arg_dict)

    return log_likelihood + log_prior