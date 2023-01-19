import numpy as np
import numba as nb
import scipy as sp
import src.utils as utils
import multiprocessing as mp
import scipy.stats as stats
import scipy.special as sp_special
import scipy.integrate as sp_integrate
from functools import partial

from NumbaQuadpack import quadpack_sig, dqags
from astropy.cosmology import Planck18 as cosmo

# ------------- PRIOR AND POP COV/RES --------------

def _log_prior(
    prior_bounds: dict, ratio_par_name: str,
    stretch_independent: bool = True, **kwargs
):

    value = 0.
    stretch_1_par = "s_1"
    stretch_2_par = "s_2"

    for value_key in kwargs.keys():
        bounds_key = ""
        # TODO: Remove 3-deep conditionals bleeeeh
        is_independent_stretch = (value_key == stretch_1_par or value_key == stretch_2_par) and stretch_independent
        is_not_ratio_par_name = value_key != ratio_par_name

        if is_independent_stretch:
            if not kwargs[stretch_1_par] < kwargs[stretch_2_par]:
                value += -np.inf
                continue
        
        if is_not_ratio_par_name:
            bounds_key = "_".join(value_key.split("_")[:-1])
        else:
            bounds_key = value_key
        
        is_in_priors = bounds_key in prior_bounds.keys()
        if is_in_priors:
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

# ---------- RB/E(B-V) PRIOR DOUBLE INTEGRAL ------------

@nb.jit
def dbl_integral_body(
    x, y, i1, i2, i3, i5,
    i6, i9, r1, r2, r3,
    Rb, gamma_Rb, lower_Rb, upper_Rb,
    Ebv, gamma_Ebv
):  

    if x < lower_Rb or x > upper_Rb:
        return 0.

    tau_Ebv = Ebv / gamma_Ebv

    # update res and cov
    r1 -= x * y * tau_Ebv
    r3 -= y * tau_Ebv

    # precalcs
    A1 = i5 * i9 - i6 * i6
    A2 = i6 * i3 - i2 * i9
    A3 = i2 * i6 - i5 * i3
    A5 = i1 * i9 - i3 * i3
    A6 = i2 * i3 - i1 * i6
    A9 = i1 * i5 - i2 * i2
    det_m1 = 1. / (i1 * A1 + i2 * A2 + i3 * A3)

    # # calculate prob
    r_inv_cov_r = det_m1 * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 + 2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    exponent_Ebv = gamma_Ebv - 1.
    value = (
        np.exp(-0.5 * r_inv_cov_r - x - y) * y**exponent_Ebv * det_m1**0.5 *
        1. / (gamma_Rb * (2*np.pi)**0.5) * np.exp(-0.5 * ((x - Rb) / gamma_Rb)**2)
    )

    return value

@nb.cfunc(quadpack_sig)
def Rb_integral(x, data):

    _data = nb.carray(data, (20,))
    i1 = _data[0]
    i2 = _data[1]
    i3 = _data[2]
    i5 = _data[4]
    i6 = _data[5]
    i9 = _data[8]
    r1 = _data[9]
    r2 = _data[10]
    r3 = _data[11]
    Rb = _data[12]
    gamma_Rb = _data[13]
    shift_Rb = _data[-4]
    lower_Rb = _data[-3]
    upper_Rb = _data[-2]
    Ebv = _data[14]
    gamma_Ebv = _data[15]
    y = _data[-1]

    return dbl_integral_body(
        x, y, i1, i2, i3, i5, 
        i6, i9, r1, r2, r3, 
        Rb, gamma_Rb, lower_Rb, upper_Rb,
        Ebv, gamma_Ebv
    )
Rb_integral_ptr = Rb_integral.address

@nb.cfunc(quadpack_sig)
def Ebv_integral(y, data):
    _data = nb.carray(data, (19,))
    _new_data = np.concatenate(
        (_data, np.array([y]))
    )

    inner_value, _, _ = dqags(
        Rb_integral_ptr, _data[-2], _data[-1], _new_data
    )

    return inner_value
Ebv_integral_ptr = Ebv_integral.address

@nb.njit
def _fast_dbl_prior_convolution(
    cov_1: np.ndarray, res_1: np.ndarray,
    cov_2: np.ndarray, res_2: np.ndarray,
    Rb_1: float, gamma_Rb_1: float, Ebv_1: float, gamma_Ebv_1: float,
    lower_bound_Ebv_1: float, upper_bound_Ebv_1: float,
    Rb_2: float, gamma_Rb_2: float, Ebv_2: float, gamma_Ebv_2: float,
    lower_bound_Ebv_2: float, upper_bound_Ebv_2: float,
    shift_Rb: float, lower_bound_Rb: float, upper_bound_Rb: float
):

    n_sn = len(cov_1)
    probs = np.zeros((n_sn, 2))
    status = np.ones((n_sn, 2), dtype='bool')
    params_1 = np.array([
        Rb_1, gamma_Rb_1,
        Ebv_1, gamma_Ebv_1,
        shift_Rb, lower_bound_Rb, upper_bound_Rb
    ])
    params_2 = np.array([
        Rb_2, gamma_Rb_2,
        Ebv_2, gamma_Ebv_2,
        shift_Rb, lower_bound_Rb, upper_bound_Rb
    ])

    for i in range(n_sn):
        tmp_params_1 = np.concatenate((
            cov_1[i].ravel(), res_1[i].ravel(), params_1
        )).copy()
        tmp_params_1.astype(np.float64)
        tmp_params_2 = np.concatenate((
            cov_2[i].ravel(), res_2[i].ravel(), params_2
        )).copy()
        tmp_params_2.astype(np.float64)
        prob_1, _, s1 = dqags(
            integrate_ptr, lower_bound_Ebv_1, upper_bound_Ebv_1, tmp_params_1
        )
        prob_2, _, s2 = dqags(
            integrate_ptr, lower_bound_Ebv_2, upper_bound_Ebv_2, tmp_params_2
        )

        probs[i, 0] = prob_1
        probs[i, 1] = prob_2
        status[i, 0] = s1
        status[i, 1] = s2

    return probs, status

def _wrapper_dbl_prior_conv(
    covs_1: np.ndarray, r_1: np.ndarray,
    Rb_1: float, gamma_Rb_1: float, Ebv_1: float, gamma_Ebv_1: float,
    covs_2: np.ndarray, r_2: np.ndarray,
    Rb_2: float, gamma_Rb_2: float, Ebv_2: float, gamma_Ebv_2: float,
    shift_Rb: float = 0., lower_bound_Rb: float = 0., upper_bound_Rb: float = 10.,
    lower_bound_Ebv: float = 0., upper_bound_Ebv: float = 10.
):

    tau_1 = Ebv_1 / gamma_Ebv_1
    tau_2 = Ebv_2 / gamma_Ebv_2
    lower_bound_Ebv_1, upper_bound_Ebv_1 = np.array([lower_bound_Ebv, upper_bound_Ebv]) / tau_1
    lower_bound_Ebv_2, upper_bound_Ebv_2 = np.array([lower_bound_Ebv, upper_bound_Ebv]) / tau_2
    norm_1 = (
        (stats.norm.cdf(upper_bound_Rb, loc=Rb_1, scale=gamma_Rb_1) - stats.norm.cdf(lower_bound_Rb, loc=Rb_1, scale=gamma_Rb_1)) *
        #sp_special.gammainc(gamma_Rb_1, upper_bound_Rb) * sp_special.gamma(gamma_Rb_1) *
        sp_special.gammainc(gamma_Ebv_1, upper_bound_Ebv_1) * sp_special.gamma(gamma_Ebv_1) * tau_1
    )
    norm_2 = (
        (stats.norm.cdf(upper_bound_Rb, loc=Rb_2, scale=gamma_Rb_2) - stats.norm.cdf(lower_bound_Rb, loc=Rb_2, scale=gamma_Rb_2)) *
        #sp_special.gammainc(gamma_Rb_2, upper_bound_Rb) * sp_special.gamma(gamma_Rb_2) *
        sp_special.gammainc(gamma_Ebv_2, upper_bound_Ebv_2) * sp_special.gamma(gamma_Ebv_2) * tau_2
    )

    probs, status = _fast_dbl_prior_convolution(
        cov_1=covs_1, res_1=r_1, cov_2=covs_2, res_2=r_2,
        Rb_1=Rb_1, gamma_Rb_1=gamma_Rb_1, Ebv_1=Ebv_1, gamma_Ebv_1=gamma_Ebv_1,
        lower_bound_Ebv_1=lower_bound_Ebv_1, upper_bound_Ebv_1=upper_bound_Ebv_1,
        Rb_2=Rb_2, gamma_Rb_2=gamma_Rb_2, Ebv_2=Ebv_2, gamma_Ebv_2=gamma_Ebv_2,
        lower_bound_Ebv_2=lower_bound_Ebv_2, upper_bound_Ebv_2=upper_bound_Ebv_2,
        shift_Rb=shift_Rb, lower_bound_Rb=lower_bound_Rb, upper_bound_Rb=upper_bound_Rb
    )
    p_1 = probs[:, 0] / norm_1
    p_2 = probs[:, 1] / norm_2

    p1_nans = np.isnan(p_1)
    p2_nans = np.isnan(p_2)

    if np.any(p1_nans):
        print("Pop 1 contains nan probabilities:", np.count_nonzero(p1_nans)/len(p1_nans)*100, "%")
        print("Pop 1 pars:", [Rb_1, gamma_Rb_1, Ebv_1, gamma_Ebv_1])
    if np.any(p2_nans):
        print("Pop 2 contains nan probabilities:", np.count_nonzero(p2_nans)/len(p2_nans)*100, "%")
        print("Pop 1 pars:", [Rb_2, gamma_Rb_2, Ebv_2, gamma_Ebv_2])
        print("Pop 2 norm:", norm_2, "\n")

    return p_1, p_2, status

# ---------- E(B-V) PRIOR INTEGRAL ------------

@nb.jit
def integral_body(
    x, i1, i2, i3, i5,
    i6, i9, r1, r2, r3,
    rb, sig_rb, Ebv, gamma_Ebv
):  

    tau_Ebv = Ebv / gamma_Ebv

    # update res and cov
    r1 -= rb * tau_Ebv * x
    r3 -= tau_Ebv * x
    i1 += sig_rb * sig_rb * tau_Ebv * tau_Ebv * x * x

    # precalcs
    exponent = gamma_Ebv - 1
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
    _data = nb.carray(data, (16,))
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
    Ebv = _data[14]
    gamma_Ebv = _data[15]
    return integral_body(
        x, i1, i2, i3, i5, i6, i9, r1, r2, r3, 
        rb, sig_rb, Ebv, gamma_Ebv
    )
integrate_ptr = integral.address

@nb.njit
def _fast_prior_convolution(
    cov_1: np.ndarray, res_1: np.ndarray,
    cov_2: np.ndarray, res_2: np.ndarray,
    rb_1: float, sig_rb_1: float, Ebv_1: float, gamma_Ebv_1: float,
    lower_bound_1: float, upper_bound_1: float,
    rb_2: float, sig_rb_2: float, Ebv_2: float, gamma_Ebv_2: float,
    lower_bound_2: float, upper_bound_2: float,
):

    n_sn = len(cov_1)
    probs = np.zeros((n_sn, 2))
    status = np.ones((n_sn, 2), dtype='bool')
    params_1 = np.array([rb_1, sig_rb_1, Ebv_1, gamma_Ebv_1])
    params_2 = np.array([rb_2, sig_rb_2, Ebv_2, gamma_Ebv_2])

    for i in range(n_sn):
        tmp_params_1 = np.concatenate((
            cov_1[i].ravel(), res_1[i].ravel(), params_1
        )).copy()
        tmp_params_1.astype(np.float64)
        tmp_params_2 = np.concatenate((
            cov_2[i].ravel(), res_2[i].ravel(), params_2
        )).copy()
        tmp_params_2.astype(np.float64)
        prob_1, _, s1 = dqags(
            integrate_ptr, lower_bound_1, upper_bound_1, tmp_params_1
        )
        prob_2, _, s2 = dqags(
            integrate_ptr, lower_bound_2, upper_bound_2, tmp_params_2
        )

        probs[i, 0] = prob_1
        probs[i, 1] = prob_2
        status[i, 0] = s1
        status[i, 1] = s2

    return probs, status

def _wrapper_prior_conv(
    covs_1: np.ndarray, r_1: np.ndarray, rb_1: float,
    sig_rb_1: float, Ebv_1: float, gamma_Ebv_1: float,
    covs_2: np.ndarray, r_2: np.ndarray, rb_2: float,
    sig_rb_2: float, Ebv_2: float, gamma_Ebv_2: float,
    lower_bound: float = 0., upper_bound: float = 10.
):

    tau_1 = Ebv_1 / gamma_Ebv_1
    tau_2 = Ebv_2 / gamma_Ebv_2
    lower_bound_1, upper_bound_1 = np.array([lower_bound, upper_bound]) / tau_1
    lower_bound_2, upper_bound_2 = np.array([lower_bound, upper_bound]) / tau_2
    norm_1 = sp_special.gammainc(gamma_Ebv_1, upper_bound_1) * sp_special.gamma(gamma_Ebv_1) * tau_1
    norm_2 = sp_special.gammainc(gamma_Ebv_2, upper_bound_2) * sp_special.gamma(gamma_Ebv_2) * tau_2

    probs, status = _fast_prior_convolution(
        covs_1, r_1, covs_2, r_2,
        rb_1=rb_1, sig_rb_1=sig_rb_1, Ebv_1=Ebv_1, gamma_Ebv_1=gamma_Ebv_1,
        lower_bound_1=lower_bound_1, upper_bound_1=upper_bound_1,
        rb_2=rb_2, sig_rb_2=sig_rb_2, Ebv_2=Ebv_2, gamma_Ebv_2=gamma_Ebv_2,
        lower_bound_2=lower_bound_2, upper_bound_2=upper_bound_2
    )

    p_1 = probs[:, 0] / norm_1
    p_2 = probs[:, 1] / norm_2

    p1_nans = np.isnan(p_1)
    p2_nans = np.isnan(p_2)

    if np.any(p1_nans):
        print("Pop 1 contains nan probabilities:", np.count_nonzero(p1_nans)/len(p1_nans)*100, "%")
        print("Pop 1 pars:", [rb_1, sig_rb_1, Ebv_1, gamma_Ebv_1])
        print("Pop 1 norm:", norm_1, "\n")
    if np.any(p2_nans):
        print("Pop 2 contains nan probabilities:", np.count_nonzero(p2_nans)/len(p2_nans)*100, "%")
        print("Pop 1 pars:", [rb_2, sig_rb_2, Ebv_2, gamma_Ebv_2])
        print("Pop 2 norm:", norm_2, "\n")

    return p_1, p_2, status

# ----------------- LOG PROB -------------------

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
    sig_c_1, sig_int_1, Rb_1, sig_Rb_1, 
    gamma_Rb_1, Ebv_1, gamma_Ebv_1,
    Mb_2, alpha_2, beta_2, s_2, sig_s_2, c_2,
    sig_c_2, sig_int_2, Rb_2, sig_Rb_2,
    gamma_Rb_2, Ebv_2, gamma_Ebv_2,
    w, H0, lower_bound_Ebv, upper_bound_Ebv,
    shift_Rb, lower_bound_Rb, upper_bound_Rb
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

    use_gaussian_Rb = (
        not gamma_Rb_1 and
        not gamma_Rb_2 and
        sig_Rb_1 and
        sig_Rb_2
    )

    if use_gaussian_Rb:
        probs_1, probs_2, status = _wrapper_prior_conv(
            covs_1=covs_1, r_1=r_1, rb_1=Rb_1, sig_rb_1=sig_Rb_1,
            Ebv_1=Ebv_1, gamma_Ebv_1=gamma_Ebv_1,
            covs_2=covs_2, r_2=r_2, rb_2=Rb_2, sig_rb_2=sig_Rb_2,
            Ebv_2=Ebv_2, gamma_Ebv_2=gamma_Ebv_2,
            lower_bound=lower_bound_Ebv, upper_bound=upper_bound_Ebv
        )
    else:
        probs_1, probs_2, status = _wrapper_dbl_prior_conv(
            covs_1=covs_1, r_1=r_1,
            Rb_1=Rb_1, gamma_Rb_1=gamma_Rb_1,
            Ebv_1=Ebv_1, gamma_Ebv_1=gamma_Ebv_1,
            covs_2=covs_2, r_2=r_2,
            Rb_2=Rb_2, gamma_Rb_2=gamma_Rb_2,
            Ebv_2=Ebv_2, gamma_Ebv_2=gamma_Ebv_2,
            shift_Rb=shift_Rb, lower_bound_Rb=lower_bound_Rb, upper_bound_Rb=upper_bound_Rb,
            lower_bound_Ebv=lower_bound_Ebv, upper_bound_Ebv=upper_bound_Ebv
        )

    # Check if any probs had non-posdef cov
    if np.any(probs_1 < 0.) | np.any(probs_2 < 0.):
        print("\nOh no, someones below 0\n")
        return -np.inf

    # TODO: Fix numerical stability by using logsumexp somehow
    log_prob = np.sum(
        np.log(
            w * probs_1 + (1-w) * probs_2
        )
    )

    if np.isnan(log_prob):
        log_prob = -np.inf

    s1, s2 = np.all(status[:, 0]), np.all(status[:, 1])
    if not s1 or not s2:
        mean1, mean2 = np.mean(probs_1), np.mean(probs_2)
        std1, std2 = np.std(probs_1), np.std(probs_2)
        f1, f2 = np.count_nonzero(~status[:, 0])/len(status), np.count_nonzero(~status[:, 1])/len(status)
        print("\nPop 1 mean and std, percentage failed:", mean1, "+-", std1, ",", f1*100, "%")
        print("Pop 2 mean and std, percentage failed:", mean2, "+-", std2, ",", f2*100, "%")
        print("Log prob:", log_prob, "\n")

    return log_prob

def generate_log_prob(
    model_cfg: dict, sn_covs: np.ndarray, 
    sn_mb: np.ndarray, sn_s: np.ndarray,
    sn_c: np.ndarray, sn_z: np.ndarray,
    lower_bound_Ebv: float, upper_bound_Ebv: float,
    shift_Rb: float, lower_bound_Rb: float, upper_bound_Rb: float
):

    init_arg_dict = {key: value for key, value in model_cfg['preset_values'].items()}
    init_arg_dict['sn_mb'] = sn_mb
    init_arg_dict['sn_s'] = sn_s
    init_arg_dict['sn_c'] = sn_c
    init_arg_dict['sn_z'] = sn_z
    init_arg_dict['sn_cov'] = sn_covs
    init_arg_dict['lower_bound_Ebv'] = lower_bound_Ebv
    init_arg_dict['upper_bound_Ebv'] = upper_bound_Ebv
    init_arg_dict['lower_bound_Rb'] = lower_bound_Rb
    init_arg_dict['upper_bound_Rb'] = upper_bound_Rb
    init_arg_dict['shift_Rb'] = shift_Rb

    global log_prob_f

    def log_prob_f(theta):

        arg_dict = {
            **init_arg_dict, **utils.theta_to_dict(
                theta=theta, shared_par_names=model_cfg['shared_par_names'], 
                independent_par_names=model_cfg['independent_par_names'],
                ratio_par_name=model_cfg['ratio_par_name']
            )
        }

        log_prior = _log_prior(
            prior_bounds=model_cfg['prior_bounds'],
            ratio_par_name=model_cfg['ratio_par_name'],
            **arg_dict
        )
        if np.isinf(log_prior):
            return log_prior
        
        if model_cfg['use_sigmoid']:
            arg_dict = utils.apply_sigmoid(
                arg_dict=arg_dict, sigmoid_cfg=model_cfg['sigmoid_cfg'],
                independent_par_names=model_cfg["independent_par_names"],
                ratio_par_name=model_cfg["ratio_par_name"]
            )

        log_likelihood = _log_likelihood(**arg_dict)

        return log_likelihood
    
    return log_prob_f