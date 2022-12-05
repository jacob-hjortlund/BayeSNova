import numpy as np
import scipy.special as sp_special
import scipy.integrate as sp_integrate
import src.utils as utils

from astropy.cosmology import Planck18 as cosmo

def _log_prior(
    prior_bounds: dict, ratio_par_name: str, **kwargs
):

    value = 0.
    for value_key in kwargs.keys():
        if value_key != ratio_par_name:
            bounds_key = "_".join(value_key.split("_")[:-1])
        else:
            bounds_key = value_key
        
        if bounds_key in prior_bounds.keys():
            value += utils.uniform(
                kwargs[value_key], **prior_bounds[bounds_key]
            )

    return value

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
    distance_modulus = cosmo.distmod(sn_z).value + 5. * np.log10(cosmo.H0.value / H0)
    r[:, 0] = sn_mb - (
        Mb + alpha * s + beta * c + distance_modulus
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

    def dust_integral(x):
        # copy arrays
        r_tmp = r.copy()
        covs_tmp = covs.copy()

        # Update residual
        r_tmp[:,0] -= rb * tau * x
        r_tmp[:,2] -= tau * x

        # Update covariances
        covs_tmp[:,0,0] += sig_rb**2 * tau**2 * x**2

        if np.any(np.linalg.det(covs_tmp) <= 0.):
            raise ValueError('Bad covs present')

        # Setup expression
        dets = np.linalg.det(covs_tmp)
        inv_covs = np.linalg.inv(covs_tmp)
        inv_det_r = np.dot(inv_covs, r_tmp.swapaxes(0,1))
        idx = np.arange(len(r_tmp))
        inv_det_r = inv_det_r[idx,:,idx]
        r_inv_det_r = np.diag(
            np.dot(r_tmp, inv_det_r.swapaxes(0,1))
        )
        values = np.exp(-0.5 * r_inv_det_r - x) * x**(alpha_g - 1.) / dets**0.5

        return values
    norm = sp_special.gammainc(alpha_g, upper_bound) * sp_special.gamma(alpha_g)
    p_convoluted = sp_integrate.quad_vec(
        dust_integral, lower_bound, upper_bound
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
        sn_cov=sn_cov, z=sn_z, alpha=alpha, beta=beta,
        sig_s=sig_s, sig_c=sig_c, sig_int=sig_int
    )
    r = _population_r(
        sn_mb=sn_mb, sn_s=sn_s, sn_c=sn_c, sn_z=sn_z,
        Mb=Mb, alpha=alpha, beta=beta, s=s, c=c, H0=H0
    )
    dust_reddening_convolved_prob = _dust_reddening_convolved_probability(
        covs=covs, r=r, rb=rb, sig_rb=sig_rb, tau=tau, alpha_g=alpha_g,
        lower_bound=lower_bound, upper_bound=upper_bound
    )

    return dust_reddening_convolved_prob

def _log_likelihood(
    sn_cov, sn_mb, sn_z, sn_s, sn_c,
    Mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, sig_int_1, Rb_1, sig_Rb_1, tau_1, alpha_g_1,
    Mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, Rb_2, sig_Rb_2, tau_2, alpha_g_2,
    w, H0
):

    pop1_probs = population_prob(
        sn_cov=sn_cov, sn_mb=sn_mb, sn_z=sn_z, sn_s=sn_s, sn_c=sn_c,
        Mb=Mb_1, alpha=alpha_1, beta=beta_1, s=s_1, sig_s=sig_s_1,
        c=c_1, sig_c=sig_c_1, sig_int=sig_int_1, rb=Rb_1, sig_rb=sig_Rb_1,
        tau=tau_1, alpha_g=alpha_g_1, H0=H0
    )

    pop2_probs = population_prob(
        sn_cov=sn_cov, sn_mb=sn_mb, sn_z=sn_z, sn_s=sn_s, sn_c=sn_c,
        Mb=Mb_2, alpha=alpha_2, beta=beta_2, s=s_2, sig_s=sig_s_2,
        c=c_2, sig_c=sig_c_2, sig_int=sig_int_2, rb=Rb_2, sig_rb=sig_Rb_2,
        tau=tau_2, alpha_g=alpha_g_2, H0=H0
    )

    # Check if any probs had non-posdef cov
    if np.any(pop1_probs < 0.) | np.any(pop2_probs < 0.):
        return -np.inf

    # TODO: Fix numerical stability by using logsumexp somehow
    log_prob = np.sum(
        np.log(
            w * pop1_probs + (1-w) * pop2_probs
        )
    )

    return log_prob

def generate_log_prob(
    model_cfg: dict, sn_covs: np.ndarray, 
    sn_mb: np.ndarray, sn_s: np.ndarray,
    sn_c: np.ndarray, sn_z: np.ndarray
):

    init_arg_dict = {key: value for key, value in model_cfg['preset_values'].items()}
    init_arg_dict['sn_mb'] = sn_mb
    init_arg_dict['sn_s'] = sn_s
    init_arg_dict['sn_c'] = sn_c
    init_arg_dict['sn_z'] = sn_z
    init_arg_dict['sn_cov'] = sn_covs

    global log_prob_f

    def log_prob_f(theta):

        arg_dict = {
            **init_arg_dict, **utils.theta_to_dict(
                theta=theta, shared_par_names=model_cfg['shared_par_names'], 
                independent_par_names=model_cfg['independent_par_names'],
                ratio_par_name=model_cfg['ratio_par_name'],
                use_sigmoid=model_cfg['use_sigmoid'],
                sigmoid_cfg=model_cfg['sigmoid_cfg']
            )
        }

        log_prior = _log_prior(
            prior_bounds=model_cfg['prior_bounds'],
            ratio_par_name=model_cfg['ratio_par_name'],
            **arg_dict
        )
        if np.isinf(log_prior):
            return log_prior
        
        log_likelihood = _log_likelihood(**arg_dict)

        return log_likelihood
    
    return log_prob_f