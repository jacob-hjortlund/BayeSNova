import numpy as np
import numba as nb
import src.utils as utils
import multiprocessing as mp
import scipy.special as sp_special
import scipy.integrate as sp_integrate

from NumbaQuadpack import quadpack_sig, dqags
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

def prior_integral(
    cov_res, rb, sig_rb, tau, alpha_g, lower_bound=1., upper_bound=10.
):

    cov = cov_res[:-1]
    res = cov_res[-1]
    exponent = alpha_g - 1.

    def f(x):
        cov_tmp = cov.copy()
        r_tmp = res.copy()

        # Update residual
        r_tmp[0] -= rb * tau * x
        r_tmp[2] -= tau * x

        # Update covariances
        cov_tmp[0,0] += sig_rb**2 * tau**2 * x**2
        
        # Setup expression
        dets = np.linalg.det(cov_tmp)
        inv_covs = np.linalg.inv(cov_tmp)
        inv_det_r = np.dot(inv_covs, r_tmp)
        r_inv_det_r = np.dot(r_tmp, inv_det_r)
        values = np.exp(-0.5 * r_inv_det_r - x) * x**exponent / dets**0.5

        return values
    
    log_p = sp_integrate.quad(f, lower_bound, upper_bound)[0]

    return log_p

def dual_pop_integration(
    cov_res: np.ndarray, pop1_prior_integral, pop2_prior_integral 
):

    cov_res_1, cov_res_2 = np.split(cov_res, 2, axis=-1)
    prob_1 = pop1_prior_integral(cov_res_1)
    prob_2 = pop2_prior_integral(cov_res_2)
    probs = np.array([[prob_1, prob_2]])

    return probs

# ---------- NUMBA EXPERIMENT ------------

@nb.jit
def integral_body(
    x, i1, i2, i3, i5,
    i6, i9, r1, r2, r3,
    rb, sig_rb, tau, alpha_g
):  
    # update res and cov
    r1 -= rb * tau * x
    r3 -= tau * x
    i1 += sig_rb * sig_rb * tau * tau * x * x

    # precalcs
    exponent = alpha_g - 1
    A1 = i5 * i9 - i6 * i6
    A2 = i6 * i3 - i2 * i9
    A3 = i2 * i6 - i5 * i3
    A5 = i1 * i9 - i3 * i3
    A6 = i2 * i3 - i1 * i6
    A9 = i1 * i5 - i2 * i2
    det_m1 = 1. / (i1 * A1 + i2 * A2 + i3 * A3)

    # # calculate prob
    r_inv_cov_r = det_m1 * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 + 2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    value = np.exp(-0.5 * r_inv_cov_r - x) * x**exponent * det_m1**0.5

    return value

@nb.cfunc(quadpack_sig)
def integral(x, data):
    _data = nb.carray(data, (13,))
    i1 = _data[0]
    i2 = _data[1]
    i3 = _data[2]
    i5 = _data[4]
    i6 = _data[5]
    i9 = _data[8]
    r1 = _data[9]
    r2 = _data[10]
    r3 = _data[11]
    rb = _data[12]
    sig_rb = _data[13]
    tau = _data[14]
    alpha_g = _data[15]
    return integral_body(
        x, i1, i2, i3, i5, i6, i9, r1, r2, r3, 
        rb, sig_rb, tau, alpha_g
    )
integrate_ptr = integral.address

@nb.njit
def _fast_prior_convolution(
    cov_1: np.ndarray, res_1: np.ndarray,
    cov_2: np.ndarray, res_2: np.ndarray,
    rb_1: float, sig_rb_1: float, tau_1: float, alpha_g_1: float,
    rb_2: float, sig_rb_2: float, tau_2: float, alpha_g_2: float,
    lower_bound: float = 0., upper_bound: float = 10.,
):

    n_sn = len(cov_1)
    probs = np.zeros((n_sn, 2))
    params_1 = np.array([rb_1, sig_rb_1, tau_1, alpha_g_1])
    params_2 = np.array([rb_2, sig_rb_2, tau_2, alpha_g_2])
    for i in range(n_sn):
        tmp_params_1 = np.concatenate((
            cov_1[i].ravel(), res_1[i].ravel(), params_1
        )).copy()
        tmp_params_1.astype(np.float64)
        tmp_params_2 = np.concatenate((
            cov_2[i].ravel(), res_2[i].ravel(), params_2
        )).copy()
        tmp_params_2.astype(np.float64)
        prob_1, _, s1, ier1 = dqags(
            integrate_ptr, lower_bound, upper_bound, tmp_params_1
        )
        prob_2, _, s2, ier2 = dqags(
            integrate_ptr, lower_bound, upper_bound, tmp_params_2
        )
        probs[i, 0] = prob_1
        probs[i, 1] = prob_2

    return probs, ier1, ier2

def _wrapper_prior_conv(
    covs_1: np.ndarray, r_1: np.ndarray, rb_1: float,
    sig_rb_1: float, tau_1: float, alpha_g_1: float,
    covs_2: np.ndarray, r_2: np.ndarray, rb_2: float,
    sig_rb_2: float, tau_2: float, alpha_g_2: float,
    lower_bound: float = 0., upper_bound: float = 10.,
    n_workers: int = 1
):
    norm_1 = sp_special.gammainc(alpha_g_1, upper_bound) * sp_special.gamma(alpha_g_1)
    norm_2 = sp_special.gammainc(alpha_g_2, upper_bound) * sp_special.gamma(alpha_g_2)

    probs, ier1, ier2 = _fast_prior_convolution(
        covs_1, r_1, covs_2, r_2,
        rb_1=rb_1, sig_rb_1=sig_rb_1, tau_1=tau_1, alpha_g_1=alpha_g_1,
        rb_2=rb_2, sig_rb_2=sig_rb_2, tau_2=tau_2, alpha_g_2=alpha_g_2,
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    p_1 = probs[:, 0] / norm_1
    p_2 = probs[:, 1] / norm_2

    return p_1, p_2

# -------- END NUMBA EXPERIMENT ----------

def _prior_convolution(
    covs_1: np.ndarray, r_1: np.ndarray, rb_1: float,
    sig_rb_1: float, tau_1: float, alpha_g_1: float,
    covs_2: np.ndarray, r_2: np.ndarray, rb_2: float,
    sig_rb_2: float, tau_2: float, alpha_g_2: float,
    lower_bound: float = 0., upper_bound: float = 10.,
    n_workers: int = 1
):

    norm_1 = sp_special.gammainc(alpha_g_1, upper_bound) * sp_special.gamma(alpha_g_1)
    norm_2 = sp_special.gammainc(alpha_g_2, upper_bound) * sp_special.gamma(alpha_g_2)
    cov_res1 = np.concatenate((covs_1, r_1[:,None]), axis=1)
    cov_res2 = np.concatenate((covs_2, r_2[:,None]), axis=1)
    iterable = np.concatenate((cov_res1, cov_res2), axis=-1)

    prior_integral_1 = utils._FunctionWrapper(
        prior_integral, args=(rb_1, sig_rb_1, tau_1, alpha_g_1, lower_bound, upper_bound)
    )
    prior_integral_2 = utils._FunctionWrapper(
        prior_integral, args=(rb_2, sig_rb_2, tau_2, alpha_g_2, lower_bound, upper_bound)
    )
    dual_pop_integral = utils._FunctionWrapper(
        dual_pop_integration, args=(prior_integral_1, prior_integral_2)
    )

    if n_workers != 1:
        with mp.Pool(n_workers) as pool:
            return_values = pool.map(dual_pop_integral, iterable)
    else:
        return_values = []
        for cov_res in iterable:
            return_values.append(dual_pop_integral(cov_res))
    
    probs = np.concatenate(return_values)
    p_1 = probs[:, 0] / norm_1
    p_2 = probs[:, 1] / norm_2

    return p_1, p_2

def _population_cov_and_residual(
    sn_cov: np.ndarray, sn_mb: np.ndarray, sn_z: np.ndarray, sn_s: np.ndarray, sn_c: np.ndarray,
    Mb: float, alpha: float, beta: float, s: float, sig_s: float,
    c: float, sig_c: float, sig_int: float, H0: float

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

    return covs, r

def _log_likelihood(
    sn_cov, sn_mb, sn_z, sn_s, sn_c,
    Mb_1, alpha_1, beta_1, s_1, sig_s_1, c_1,
    sig_c_1, sig_int_1, Rb_1, sig_Rb_1, tau_1, alpha_g_1,
    Mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, Rb_2, sig_Rb_2, tau_2, alpha_g_2,
    w, H0, lower_bound, upper_bound, n_workers
):

    covs_1, r_1 = _population_cov_and_residual(
        sn_cov=sn_cov, sn_mb=sn_mb, sn_z=sn_z, sn_s=sn_s,
        sn_c=sn_c, Mb=Mb_1, alpha=alpha_1, beta=beta_1, s=s_1,
        sig_s=sig_s_1, c=c_1, sig_c=sig_c_1, sig_int=sig_int_1, H0=H0
    )

    covs_2, r_2 = _population_cov_and_residual(
        sn_cov=sn_cov, sn_mb=sn_mb, sn_z=sn_z, sn_s=sn_s,
        sn_c=sn_c, Mb=Mb_2, alpha=alpha_2, beta=beta_2, s=s_2,
        sig_s=sig_s_2, c=c_2, sig_c=sig_c_2, sig_int=sig_int_2, H0=H0
    )

    probs_1, probs_2 = _wrapper_prior_conv(
        covs_1=covs_1, r_1=r_1, rb_1=Rb_1, sig_rb_1=sig_Rb_1,
        tau_1=tau_1, alpha_g_1=alpha_g_1,
        covs_2=covs_2, r_2=r_2, rb_2=Rb_2, sig_rb_2=sig_Rb_2,
        tau_2=tau_2, alpha_g_2=alpha_g_2,
        lower_bound=lower_bound, upper_bound=upper_bound, n_workers=n_workers
    )

    # Check if any probs had non-posdef cov
    if np.any(probs_1 < 0.) | np.any(probs_2 < 0.):
        print("Oh no, someones below 0")
        return -np.inf

    # TODO: Fix numerical stability by using logsumexp somehow
    log_prob = np.sum(
        np.log(
            w * probs_1 + (1-w) * probs_2
        )
    )

    return log_prob

def generate_log_prob(
    model_cfg: dict, sn_covs: np.ndarray, 
    sn_mb: np.ndarray, sn_s: np.ndarray,
    sn_c: np.ndarray, sn_z: np.ndarray,
    lower_bound: float, upper_bound: float,
    n_workers: int
):

    init_arg_dict = {key: value for key, value in model_cfg['preset_values'].items()}
    init_arg_dict['sn_mb'] = sn_mb
    init_arg_dict['sn_s'] = sn_s
    init_arg_dict['sn_c'] = sn_c
    init_arg_dict['sn_z'] = sn_z
    init_arg_dict['sn_cov'] = sn_covs
    init_arg_dict['lower_bound'] = lower_bound
    init_arg_dict['upper_bound'] = upper_bound
    init_arg_dict['n_workers'] = n_workers

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