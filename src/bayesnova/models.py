import inspect
import warnings
import numpy as np
import numba as nb
import bayesnova.utils as utils
import scipy.stats as stats
import scipy.special as sp_special
import astropy.cosmology as apy_cosmo

from functools import partial
from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags, ldqag

import bayesnova.preprocessing as prep
import bayesnova.cosmology_utils as cosmo_utils

NULL_VALUE = -9999.0
H0_CONVERSION_FACTOR = 0.001022
DH_70 = 4282.7494

# ---------- E(B-V) PRIOR INTEGRAL ------------

@nb.jit()
def Ebv_integral_body(
    x, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, r1, r2, r3,
    selection_bias_correction,
    rb, sig_rb, Ebv, tau_Ebv, gamma_Ebv,
):  

    if tau_Ebv == NULL_VALUE:
        tau_Ebv = Ebv / gamma_Ebv

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
    r_inv_cov_r = 1./det * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 + 2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    value = np.exp(-0.5 * r_inv_cov_r - x + exponent * np.log(x) - 0.5 * logdet - 0.5 * np.log(2 * np.pi))

    return value

@nb.cfunc(quadpack_sig)
def Ebv_integral(x, data):
    _data = nb.carray(data, (18,))
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
    rb = _data[13]
    sig_rb = _data[14]
    Ebv = _data[15]
    tau_Ebv = _data[16]
    gamma_Ebv = _data[17]
    return Ebv_integral_body(
        x, i1, i2, i3, i4, i5, i6, i7, i8, i9, r1, r2, r3,
        selection_bias_correction,
        rb, sig_rb, Ebv, tau_Ebv, gamma_Ebv
    )
Ebv_integral_ptr = Ebv_integral.address

@nb.njit
def Ebv_prior_convolution(
    cov_1: np.ndarray, res_1: np.ndarray,
    cov_2: np.ndarray, res_2: np.ndarray,
    Rb_1: float, Rb_2: float,
    sig_Rb_1: float, sig_Rb_2: float,
    Ebv_1: float, Ebv_2: float,
    tau_Ebv_1: float, tau_Ebv_2: float,
    gamma_Ebv_1: float, gamma_Ebv_2: float,
    lower_bound_Ebv_1: float, lower_bound_Ebv_2: float,
    upper_bound_Ebv_1: float, upper_bound_Ebv_2: float,
    selection_bias_correction: np.ndarray,
):

    n_sn = len(cov_1)
    logprobs = np.zeros((n_sn, 2))
    status = np.ones((n_sn, 2), dtype='bool')
    params_1 = np.array([
        Rb_1, sig_Rb_1, Ebv_1, tau_Ebv_1, gamma_Ebv_1
    ])
    params_2 = np.array([
        Rb_2, sig_Rb_2, Ebv_2, tau_Ebv_2, gamma_Ebv_2
    ])
        
    for i in range(n_sn):
        bias_corr = np.array([selection_bias_correction[i]])
        tmp_params_1 = np.concatenate((
            cov_1[i].ravel(), res_1[i].ravel(),
            bias_corr, params_1
        )).copy()
        tmp_params_1.astype(np.float64)
        tmp_params_2 = np.concatenate((
            cov_2[i].ravel(), res_2[i].ravel(),
            bias_corr, params_2
        )).copy()
        tmp_params_2.astype(np.float64)
        
        logprob_1, _, s1, _ = dqags(
            funcptr=Ebv_integral_ptr, a=lower_bound_Ebv_1,
            b=upper_bound_Ebv_1, data=tmp_params_1
        )

        logprob_2, _, s2, _ = dqags(
            funcptr=Ebv_integral_ptr, a=lower_bound_Ebv_2,
            b=upper_bound_Ebv_2, data=tmp_params_2
        )

        logprobs[i, 0] = logprob_1
        logprobs[i, 1] = logprob_2
        status[i, 0] = s1
        status[i, 1] = s2

    return logprobs, status

# ---------- LOG E(B-V) PRIOR INTEGRAL ------------

@nb.jit()
def Ebv_integral_log_body(
    x, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, r1, r2, r3,
    selection_bias_correction,
    rb, sig_rb, Ebv, tau_Ebv, gamma_Ebv,
):  

    if tau_Ebv == NULL_VALUE:
        tau_Ebv = Ebv / gamma_Ebv

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
    r_inv_cov_r = 1./det * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 + 2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    value = -0.5 * r_inv_cov_r - x + exponent * np.log(x) - 0.5 * logdet - 0.5 * np.log(2 * np.pi)

    return value

@nb.cfunc(quadpack_sig)
def Ebv_log_integral(x, data):
    _data = nb.carray(data, (18,))
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
    rb = _data[13]
    sig_rb = _data[14]
    Ebv = _data[15]
    tau_Ebv = _data[16]
    gamma_Ebv = _data[17]
    return Ebv_integral_log_body(
        x, i1, i2, i3, i4, i5, i6, i7, i8, i9, r1, r2, r3,
        selection_bias_correction,
        rb, sig_rb, Ebv, tau_Ebv, gamma_Ebv
    )
Ebv_log_integral_ptr = Ebv_log_integral.address

@nb.njit
def Ebv_prior_log_convolution(
    cov_1: np.ndarray, res_1: np.ndarray,
    cov_2: np.ndarray, res_2: np.ndarray,
    Rb_1: float, Rb_2: float,
    sig_Rb_1: float, sig_Rb_2: float,
    Ebv_1: float, Ebv_2: float,
    tau_Ebv_1: float, tau_Ebv_2: float,
    gamma_Ebv_1: float, gamma_Ebv_2: float,
    lower_bound_Ebv_1: float, lower_bound_Ebv_2: float,
    upper_bound_Ebv_1: float, upper_bound_Ebv_2: float,
    selection_bias_correction: np.ndarray,
):

    n_sn = len(cov_1)
    logprobs = np.zeros((n_sn, 2))
    status = np.ones((n_sn, 2), dtype='bool')
    params_1 = np.array([
        Rb_1, sig_Rb_1, Ebv_1, tau_Ebv_1, gamma_Ebv_1
    ])
    params_2 = np.array([
        Rb_2, sig_Rb_2, Ebv_2, tau_Ebv_2, gamma_Ebv_2
    ])
        
    for i in range(n_sn):
        bias_corr = np.array([selection_bias_correction[i]])
        tmp_params_1 = np.concatenate((
            cov_1[i].ravel(), res_1[i].ravel(),
            bias_corr, params_1
        )).copy()
        tmp_params_1.astype(np.float64)
        tmp_params_2 = np.concatenate((
            cov_2[i].ravel(), res_2[i].ravel(),
            bias_corr, params_2
        )).copy()
        tmp_params_2.astype(np.float64)
        
        logprob_1, _, s1, _ = ldqag(
            funcptr=Ebv_log_integral_ptr, a=lower_bound_Ebv_1,
            b=upper_bound_Ebv_1, data=tmp_params_1
        )

        logprob_2, _, s2, _ = ldqag(
            funcptr=Ebv_log_integral_ptr, a=lower_bound_Ebv_2,
            b=upper_bound_Ebv_2, data=tmp_params_2
        )

        logprobs[i, 0] = logprob_1
        logprobs[i, 1] = logprob_2
        status[i, 0] = s1
        status[i, 1] = s2

    return logprobs, status
