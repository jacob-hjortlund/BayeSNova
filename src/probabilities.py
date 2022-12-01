import numpy as np
import scipy.special as sp_special
import scipy.integrate as sp_integrate
import src.utils as utils

from astropy.cosmology import Planck18_arXiv_v2 as cosmo

def _log_prior(
    prior_bounds, **kwargs
):

    for param, key in prior_bounds:
        print(param, key)

    return 1

def _population_covariance(
    sn_cov: np.ndarray, z: np.ndarray, alpha: float,
    beta: float, sig_s: float, sig_c: float, sig_int: float
) -> np.ndarray:
    """Calculate the covariance matrix shared by all SN in a given population

    Args:
        sn_cov (np.ndarray): SN covariance matrices
        z (np.ndarray): SN redshifts
        alpha (float): Population stretch correction
        beta (float): Population intrinsic color correction
        sig_s (float): Population stretch uncertainty
        sig_c (float): Population intrinsic color uncertainty
        sig_int (float): Population intrinsic scatter

    Returns:
        np.ndarray: Shared covariance matrix
    """

    cov = np.zeros(sn_cov.shape)
    disp_v_pec = 200. # km / s
    c = 300000. # km / s
    cov[:,0,0] = sig_int**2 + alpha**2 * sig_s**2 + beta**2 * sig_c**2
    cov[:,0,0] += z**(-2) * (
        (5. / np.log(10.))**2
        * (disp_v_pec / c)**2
    )
    cov[:,1,1] = sig_s**2
    cov[:,2,2] = sig_c**2
    cov[:,0,1] = alpha * sig_s**2
    cov[:,0,2] = beta * sig_c**2
    cov[:,1,0] = cov[:,0,1]
    cov[:,2,0] = cov[:,0,2]

    cov += sn_cov

    return cov

def _population_r(
    sn_mb: np.ndarray, sn_s: np.ndarray, sn_c: np.ndarray, sn_z: np.ndarray,
    Mb: float, alpha: float, beta: float, s: float, c: float, H0: float,
) -> np.ndarray:
    """Calculate residual between data and mean vector

    Args:
        sn_mb (np.ndarray): SN apparent magnitudes
        sn_s (np.ndarray): SN stretches
        sn_c (np.ndarray): SN intrinsic colors
        sn_z (np.ndarray): SN redshifts
        Mb (float): Population absolute magnitude
        alpha (float): Population stretch correction
        beta (float): Population intrinsic color correction
        s (float): Population stretch
        c (float): Population intrinsic color
        H0 (float): Hubble constant

    Returns:
        np.ndarray: Residuals
    """

    r = np.zeros((len(sn_mb), 3))
    distance_modulus = cosmo.distmod(sn_z) + 5. * np.log10(cosmo.H0.value / H0)
    r[:, 0] = sn_mb - (
        Mb + cosmo.distmod(sn_z) + alpha * s + beta * c + distance_modulus
    ) 
    r[:, 1] = sn_s - s
    r[:, 2] = sn_c - c

    return r

def _dust_reddening_convolved_probability(
    covs: np.ndarray, r: np.ndarray, rb: float,
    sig_rb: float, tau: float, alpha_g: float,
    lower_bound: float = 0., upper_bound: float = 10.
) -> np.ndarray:
    """Vectorized numerical convolution of partial posterior and
    dust reddening gamma distribution.

    Args:
        covs (np.ndarray): Covariance matrices with shape (N,3,3)
        r (np.ndarray): Residuals with shape (N,3)
        rb (float): Population extinction coefficient
        sig_rb (float): Population extinction coefficient scatter
        tau (float): Population extinction dist scale parameter
        alpha_g (float): Population extinction dist shape parameter
        lower_bound (float, optional): Lower bound on integral. Defaults to 0.
        upper_bound (float, optional): Upper bound on integral. Defaults to 10.

    Returns:
        np.ndarray: _description_
    """

    def f(x):
        # Update residual
        r[:,0] -= rb * tau * x
        r[:,2] -= tau * x

        # Update covariances
        covs[:,0,0] += sig_rb**2 * tau**2 * x**2

        # Setup expression
        dets = np.linalg.det(covs)
        inv_covs = np.linalg.inv(covs)
        inv_det_r = np.dot(inv_covs, r.swapaxes(0,1))
        r_inv_det_r = np.diag(
            np.dot(r, inv_det_r.swapaxes(0,1))
        )
        values = np.exp(-0.5 * r_inv_det_r - x) * x**(alpha_g - 1.) / dets**2

        return values

    norm = sp_special.gammainc(alpha_g, upper_bound) * sp_special.gamma(alpha_g)
    p_convoluted = sp_integrate.quad_vec(
        f, lower_bound, upper_bound
    )[0] / norm
        
    return p_convoluted

def population_prob(
    sn_cov: np.ndarray, sn_mb: np.ndarray, sn_z: np.ndarray, sn_s: np.ndarray, sn_c: np.ndarray,
    Mb: float, alpha: float, beta: float, s: float, sig_s: float,
    c: float, sig_c: float, sig_int: float, rb: float, sig_rb: float,
    tau: float, alpha_g: float, H0: float, lower_bound: float = 0., upper_bound: float = 10.

) -> np.ndarray:
    """Calculate convolved probabilities for given population distribution

    Args:
        sn_cov (np.ndarray): SN covariance matrices
        sn_mb (np.ndarray): SN apparent magnitudes
        sn_z (np.ndarray): SN redshifts
        sn_s (np.ndarray): SN stretch parameters
        sn_c (np.ndarray): SN intrinsic colour
        Mb (float): Population absolute magnitudes
        alpha (float): Population stretch correction
        beta (float): Population intrinsic colour correction
        s (float): Population stretch parameter
        sig_s (float): Population stretch parameter scatter
        c (float): Population intrinsic colour
        sig_c (float): Population intrinsic colour scatter
        sig_int (float): Population intrinsic scatter
        rb (float): Population extinction coefficient
        sig_rb (float): Population extinction coefficient scatter
        tau (float): Population dust reddening dist scale
        alpha_g (float): Population dust reddening dist shapeparameter
        H0 (float): Hubble constant
        lower_bound (float, optional): Dust reddening convolution lower bound. Defaults to 0.
        upper_bound (float, optional): Dust reddening convolution lower bound. Defaults to 10.

    Returns:
        np.ndarray: Convolved population probabilities
    """
    
    covs = _population_covariance(
        sn_cov, sn_z, alpha, beta, sig_s, sig_c, sig_int
    )
    r = _population_r(
        sn_mb, sn_s, sn_c, sn_z, Mb, alpha, beta, s, c, H0
    )
    dust_reddening_convolved_prob = _dust_reddening_convolved_probability(
        covs, r, rb, sig_rb, tau, alpha_g, lower_bound, upper_bound
    )

    return dust_reddening_convolved_prob

def _log_likelihood(
    sn_cov, sn_mb, sn_z, sn_s, sn_c,
    Mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, sig_int_1, rb_1, sig_rb_1, tau_1, alpha_g_1,
    Mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, rb_2, sig_rb_2, tau_2, alpha_g_2,
    w, H0
):

    pop1_probs = population_prob(
        sn_cov, sn_mb, sn_z, sn_s, sn_c, Mb_1, alpha_1, beta_1, s_1, sig_s_1,
        c_1, sig_c_1, sig_int_1, rb_1, sig_rb_1, tau_1, alpha_g_1, H0
    )

    pop2_probs = population_prob(
        sn_cov, sn_mb, sn_z, sn_s, sn_c, Mb_2, alpha_2, beta_2, s_2, sig_s_2,
        c_2, sig_c_2, sig_int_2, rb_2, sig_rb_2, tau_2, alpha_g_2, H0
    )

    # Check if any probs had non-posdef cov
    if np.any(pop1_probs < 0.) | np.any(pop2_probs < 0.):
        return -np.inf

    log_prob = np.sum(
        np.log(w * pop1_probs + (1-w) * pop2_probs)
    )

    return log_prob

def generate_log_prob(
    shared_par_names: list, independent_par_names: list, ratio_par_name: list, prior_bounds: dict,
    sn_covariances: np.ndarray, mb: np.ndarray, s: np.ndarray, c: np.ndarray, z: np.ndarray
):
    def log_prob_f(theta):

        arg_dict = utils.theta_to_dict(
            theta, shared_par_names, independent_par_names, ratio_par_name 
        )
        arg_dict['sn_mb'] = mb
        arg_dict['sn_s'] = s
        arg_dict['sn_c'] = c
        arg_dict['sn_z'] = z
        arg_dict['sn_cov'] = sn_covariances

        log_prior = _log_prior(prior_bounds, **arg_dict)
        if np.isinf(log_prior):
            return log_prior
        
        log_likelihood = _log_likelihood(**arg_dict)

        return log_likelihood + log_prior
    
    return log_prob_f