import numpy as np
import scipy.special as sp_special
import scipy.integrate as sp_integrate
import src.utils as utils

from astropy.cosmology import Planck18_arXiv_v2 as cosmo

def _log_prior(
    Mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, sig_int_1, rb_1, sig_rb_1, tau_1, alpha_g_1,
    Mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, rb_2, sig_rb_2, tau_2, alpha_g_2
):

    return 1

def _population_covariance(
    z: np.ndarray, alpha: float, beta: float,
    sig_s: float, sig_c: float, sig_int: float
) -> np.ndarray:
    """Calculate the covariance matrix shared by all SN in a given population

    Args:
        z (np.ndarray): SN redshifts
        alpha (float): Stretch correction
        beta (float): Intrinsic color correction
        sig_s (float): Stretch uncertainty
        sig_c (float): Intrinsic color uncertainty
        sig_int (float): Intrinsic scatter

    Returns:
        np.ndarray: Shared covariance matrix
    """

    cov = np.zeros((len(z),3,3))
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

    return cov

def _population_r(
    mb: np.ndarray, s: np.ndarray, c: np.ndarray, z: np.ndarray, Mb: float,
    alpha_h: float, beta_h: float, s_h: float, c_h: float, H0: float,
) -> np.ndarray:
    """Calculate residual between data and mean vector

    Args:
        mb (np.ndarray): Apparent magnitudes
        s (np.ndarray): SN stretches
        c (np.ndarray): SN intrinsic colors
        z (np.ndarray): SN redshifts
        Mb (float): Population absolute magnitude
        alpha_h (float): Population stretch correction
        beta_h (float): Population intrinsic color correction
        s_h (float): Population stretch
        c_h (float): Population intrinsic color
        H0 (float): Hubble constant

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
    mb: np.ndarray, z: np.ndarray, s: np.ndarray, c: np.ndarray,
    Mb_h: float, alpha_h: float, beta_h: float, s_h: float, sig_s_h: float,
    c_h: float, sig_c_h: float, sig_int: float, rb_h: float, sig_rb_h: float,
    tau_h: float, alpha_g: float, H0: float, lower_bound: float = 0., upper_bound: float = 10.

) -> np.ndarray:
    """Calculate convolved probabilities for given population distribution

    Args:
        mb (np.ndarray): _description_
        z (np.ndarray): _description_
        s (np.ndarray): _description_
        c (np.ndarray): _description_
        Mb_h (float): _description_
        alpha_h (float): _description_
        beta_h (float): _description_
        s_h (float): _description_
        sig_s_h (float): _description_
        c_h (float): _description_
        sig_c_h (float): _description_
        sig_int (float): _description_
        rb_h (float): _description_
        sig_rb_h (float): _description_
        tau_h (float): _description_
        alpha_g (float): _description_
        H0 (float): _description_
        lower_bound (float, optional): _description_. Defaults to 0..
        upper_bound (float, optional): _description_. Defaults to 10..

    Returns:
        _type_: _description_
    """
    
    covs = _population_covariance(
        z, alpha_h, beta_h, sig_s_h, sig_c_h, sig_int
    )
    r = _population_r(
        mb, s, c, z, Mb_h, alpha_h, beta_h, s_h, c_h, H0
    )
    dust_reddening_convolved_prob = _dust_reddening_convolved_probability(
        covs, r, rb_h, sig_rb_h, tau_h, alpha_g, lower_bound, upper_bound
    )

    return dust_reddening_convolved_prob

def _log_likelihood(
    mb, z, s, c,
    Mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, sig_int_1, rb_1, sig_rb_1, tau_1, alpha_g_1,
    Mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, rb_2, sig_rb_2, tau_2, alpha_g_2,
    w, H0
):

    pop1_probs = population_prob(
        mb, z, s, c, Mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1, 
        sig_c_1, sig_int_1, rb_1, sig_rb_1, tau_1, alpha_g_1, H0
    )

    pop2_probs = population_prob(
        mb, z, s, c, Mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2, 
        sig_c_2, sig_int_2, rb_2, sig_rb_2, tau_2, alpha_g_2, H0
    )

    if np.any(pop1_probs < 0.) | np.any(pop2_probs < 0.):
        return -np.inf

    log_prob = np.sum(
        np.log(w * pop1_probs + (1-w) * pop2_probs)
    )

    return log_prob

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