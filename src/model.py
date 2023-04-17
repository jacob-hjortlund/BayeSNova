import warnings
import numpy as np
import numba as nb
import src.utils as utils
import scipy.special as sp_special
import astropy.cosmology as apy_cosmo

from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags

import src.preprocessing as prep
import src.cosmology_utils as cosmo_utils

NULL_VALUE = -9999.0
H0_CONVERSION_FACTOR = 0.001022
DH_70 = 4282.7494

# ---------- E(B-V) PRIOR INTEGRAL ------------

@nb.jit
def Ebv_integral_body(
    x, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, r1, r2, r3,
    selection_bias_correction,
    rb, sig_rb, Ebv, tau_Ebv, gamma_Ebv
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
    det_m1 = 1. / (i1 * A1 + i2 * A2 + i3 * A3)

    if det_m1 < 0:
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
        det_m1 = 1. / (i1 * A1 + i2 * A2 + i3 * A3)

    # # calculate prob
    r_inv_cov_r = det_m1 * (r1 * r1 * A1 + r2 * r2 * A5 + r3 * r3 * A9 + 2 * (r1 * r2 * A2 + r1 * r3 * A3 + r2 * r3 * A6))
    value = np.exp(-0.5 * r_inv_cov_r - x) * x**exponent * det_m1**0.5

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
def _Ebv_prior_convolution(
    cov_1: np.ndarray, res_1: np.ndarray,
    cov_2: np.ndarray, res_2: np.ndarray,
    Rb_1: float, Rb_2: float,
    sig_Rb_1: float, sig_Rb_2: float,
    tau_Rb_1: float, tau_Rb_2: float,
    gamma_Rb_1: float, gamma_Rb_2: float,
    Ebv_1: float, Ebv_2: float,
    tau_Ebv_1: float, tau_Ebv_2: float,
    gamma_Ebv_1: float, gamma_Ebv_2: float,
    lower_bound_Rb_1: float, lower_bound_Rb_2: float,
    upper_bound_Rb_1: float, upper_bound_Rb_2: float,
    lower_bound_Ebv_1: float, lower_bound_Ebv_2: float,
    upper_bound_Ebv_1: float, upper_bound_Ebv_2: float,
    shift_Rb: float, selection_bias_correction: np.ndarray,
):

    n_sn = len(cov_1)
    probs = np.zeros((n_sn, 2))
    status = np.ones((n_sn, 2), dtype='bool')
    params_1 = np.array([Rb_1, sig_Rb_1, Ebv_1, tau_Ebv_1, gamma_Ebv_1])
    params_2 = np.array([Rb_2, sig_Rb_2, Ebv_2, tau_Ebv_2, gamma_Ebv_2])

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
        
        prob_1, _, s1 = dqags(
            Ebv_integral_ptr, lower_bound_Ebv_1, upper_bound_Ebv_1, tmp_params_1
        )

        prob_2, _, s2 = dqags(
            Ebv_integral_ptr, lower_bound_Ebv_2, upper_bound_Ebv_2, tmp_params_2
        )

        probs[i, 0] = prob_1
        probs[i, 1] = prob_2
        status[i, 0] = s1
        status[i, 1] = s2

    return probs, status

# ---------- RB/E(B-V) PRIOR DOUBLE INTEGRAL ------------

@nb.jit
def dbl_integral_body(
    x, y, i1, i2, i3, i5,
    i6, i9, r1, r2, r3,
    Rb, tau_Rb, gamma_Rb, shift_Rb,
    Ebv, tau_Ebv, gamma_Ebv
):

    if tau_Ebv == NULL_VALUE:
        tau_Ebv = Ebv / gamma_Ebv
    if tau_Rb == NULL_VALUE:
        tau_Rb = Rb / gamma_Rb

    if x < 0.:
        return 0.

    # update res and cov
    r1 -= (x * tau_Rb + shift_Rb) * y * tau_Ebv
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
    exponent_Rb = gamma_Rb - 1.
    value = (
        np.exp(-0.5 * r_inv_cov_r - x - y) * x**exponent_Rb * y**exponent_Ebv * det_m1**0.5 
    )

    return value

@nb.cfunc(quadpack_sig)
def Rb_integral(x, data):

    _data = nb.carray(data, (22,))
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
    tau_Rb = _data[13]
    gamma_Rb = _data[14]
    Ebv = _data[15]
    tau_Ebv = _data[16]
    gamma_Ebv = _data[17]
    shift_Rb = _data[-4]
    y = _data[-1]

    return dbl_integral_body(
        x, y, i1, i2, i3, i5, 
        i6, i9, r1, r2, r3, 
        Rb, tau_Rb, gamma_Rb, shift_Rb,
        Ebv, tau_Ebv, gamma_Ebv
    )
Rb_integral_ptr = Rb_integral.address

@nb.cfunc(quadpack_sig)
def Ebv_Rb_integral(y, data):
    _data = nb.carray(data, (21,))
    _new_data = np.concatenate(
        (_data, np.array([y]))
    )

    inner_value, _, _  = dqags(
        Rb_integral_ptr, _data[-2], _data[-1], _new_data
    )

    return inner_value
Ebv_Rb_integral_ptr = Ebv_Rb_integral.address

@nb.njit
def _Ebv_Rb_prior_convolution(
    cov_1: np.ndarray, res_1: np.ndarray,
    cov_2: np.ndarray, res_2: np.ndarray,
    Rb_1: float, Rb_2: float,
    sig_Rb_1: float, sig_Rb_2: float,
    tau_Rb_1: float, tau_Rb_2: float,
    gamma_Rb_1: float, gamma_Rb_2: float,
    Ebv_1: float, Ebv_2: float,
    tau_Ebv_1: float, tau_Ebv_2: float,
    gamma_Ebv_1: float, gamma_Ebv_2: float,
    lower_bound_Rb_1: float, lower_bound_Rb_2: float,
    upper_bound_Rb_1: float, upper_bound_Rb_2: float,
    lower_bound_Ebv_1: float, lower_bound_Ebv_2: float,
    upper_bound_Ebv_1: float, upper_bound_Ebv_2: float,
    shift_Rb: float
):

    n_sn = len(cov_1)
    probs = np.zeros((n_sn, 2))
    status = np.ones((n_sn, 2), dtype='bool')
    params_1 = np.array([
        Rb_1, tau_Rb_1, gamma_Rb_1,
        Ebv_1, tau_Ebv_1, gamma_Ebv_1,
        shift_Rb, lower_bound_Rb_1, upper_bound_Rb_1
    ])
    params_2 = np.array([
        Rb_2, tau_Rb_2, gamma_Rb_2,
        Ebv_2, tau_Ebv_2, gamma_Ebv_2,
        shift_Rb, lower_bound_Rb_2, upper_bound_Rb_2
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
            Ebv_Rb_integral_ptr, lower_bound_Ebv_1, upper_bound_Ebv_1, tmp_params_1
        )

        prob_2, _, s2 = dqags(
            Ebv_Rb_integral_ptr, lower_bound_Ebv_2, upper_bound_Ebv_2, tmp_params_2
        )

        probs[i, 0] = prob_1
        probs[i, 1] = prob_2
        status[i, 0] = s1
        status[i, 1] = s2

    return probs, status

# ---------- MODEL CLASS ------------

class Model():

    def __init__(self) -> None:
        pass

    def log_prior(self, par_dict: dict) -> float:
        
        value = 0.
        stretch_1_par = prep.global_model_cfg.stretch_par_name + "_1"
        stretch_2_par = prep.global_model_cfg.stretch_par_name + "_2"
        if prep.global_model_cfg.stretch_par_name in prep.global_model_cfg.independent_par_names:
            stretch_independent = True
        else:
            stretch_independent = False

        for value_key in par_dict.keys():
            
            skip_this_par = np.all(par_dict[value_key] == NULL_VALUE)
            if skip_this_par:
                continue
            
            if value_key == 'host_galaxy_means':
                if np.any(
                    (par_dict[value_key] <= -20) | (par_dict[value_key] >= 20)
                ):
                    value += -np.inf
                    break

            if value_key == 'host_galaxy_sigs':
                if np.any(
                    (par_dict[value_key] < 0.) | (par_dict[value_key] >= 20.)
                ):
                    value += -np.inf
                    break
            
            # TODO: Remove 3-deep conditionals bleeeeh
            bounds_key = ""
            is_independent_stretch = (
                (
                    value_key == stretch_1_par or value_key == stretch_2_par
                ) and stretch_independent
            )

            is_not_ratio_par_name = value_key != prep.global_model_cfg.ratio_par_name
            is_not_cosmology_par_name = not value_key in prep.global_model_cfg.cosmology_par_names
            split_value_key = is_not_ratio_par_name and is_not_cosmology_par_name

            if is_independent_stretch:
                if not par_dict[stretch_1_par] < par_dict[stretch_2_par]:
                    value += -np.inf
                    break
            
            if split_value_key:
                bounds_key = "_".join(value_key.split("_")[:-1])
            else:
                bounds_key = value_key
            
            is_in_priors = bounds_key in prep.global_model_cfg.prior_bounds.keys()
            if is_in_priors:
                par_value = par_dict[value_key]
                is_dtd_rate = bounds_key == 'eta'
                if is_dtd_rate:
                    par_value = np.log10(par_value)
                value += utils.uniform(
                    par_value, **prep.global_model_cfg.prior_bounds[bounds_key]
                )

            if np.isinf(value):
                break

        return value
    
    def population_covariances(
        self,
        alpha_1: float, alpha_2: float,
        beta_1: float, beta_2: float,
        sig_int_1: float, sig_int_2: float,
        sig_s_1: float, sig_s_2: float,
        sig_c_1: float, sig_c_2: float,
    ) -> np.ndarray:

        z = prep.sn_observables[:, 3]

        cov = np.tile(prep.sn_covariances, (2, 1, 1, 1))
        disp_v_pec = 200. # km / s
        c = 300000. # km / s
        
        cov[:,:,0,0] += np.array([
            [sig_int_1**2 + alpha_1**2 * sig_s_1**2 + beta_1**2 * sig_c_1**2],
            [sig_int_2**2 + alpha_2**2 * sig_s_2**2 + beta_2**2 * sig_c_2**2]
        ])

        distmod_var = z**(-2) * (
            (5. / np.log(10.))**2
            * (disp_v_pec / c)**2
        )
        distmod_var[prep.idx_calibrator_sn] = 0.
        cov[:,:,0,0] += np.tile(
            distmod_var, (2, 1)
        )

        cov[:,:,1,1] += np.array([
            [sig_s_1**2], [sig_s_2**2]
        ])
        cov[:,:,2,2] += np.array([
            [sig_c_1**2], [sig_c_2**2]
        ])
        cov[:,:,0,1] += np.array([
            [alpha_1 * sig_s_1**2], [alpha_2 * sig_s_2**2]
        ])
        cov[:,:,0,2] += np.array([
            [beta_1 * sig_c_1**2], [beta_2 * sig_c_2**2]
        ])
        cov[:,:,1,0] = cov[:,:,0,1]
        cov[:,:,2,0] = cov[:,:,0,2]

        return cov[0], cov[1]

    def population_residuals(
        self,
        Mb_1: float, Mb_2: float,
        s_1: float, s_2: float,
        c_1: float, c_2: float,
        alpha_1: float, alpha_2: float,
        beta_1: float, beta_2: float,
        cosmo: apy_cosmo.Cosmology
    ) -> np.ndarray:
        
        #global sn_observables
        mb = prep.sn_observables[:, 0]
        s = prep.sn_observables[:, 1]
        c = prep.sn_observables[:, 2]
        z = prep.sn_observables[:, 3]

        residuals = np.zeros((2, len(mb), 3))

        distmod_values = cosmo.distmod(z).value
        distmod_values[prep.idx_calibrator_sn] = prep.calibrator_distance_moduli
        distance_moduli = np.tile(
            distmod_values, (2, 1)
        )

        residuals[:, :, 0] = np.tile(mb, (2, 1)) - np.array([
            [Mb_1 + alpha_1 * s_1 + beta_1 * c_1],
            [Mb_2 + alpha_2 * s_2 + beta_2 * c_2]
        ]) - distance_moduli
        residuals[:, :, 1] = np.tile(s, (2, 1)) - np.array([
            [s_1], [s_2]
        ])
        residuals[:, :, 2] = np.tile(c, (2, 1)) - np.array([
            [c_1], [c_2]
        ])

        return residuals[0], residuals[1]

    def get_upper_bounds(
        self, quantiles: np.ndarray, gamma_1: float, gamma_2: float, par: str
    ) -> tuple:
        
        if (
            quantiles is not None and
            gamma_1 != NULL_VALUE and
            gamma_2 != NULL_VALUE
        ):
            idx_upper_bound_1 = utils.find_nearest_idx(quantiles[0], gamma_1)
            idx_upper_bound_2 = utils.find_nearest_idx(quantiles[0], gamma_2)
            upper_bound_1 = quantiles[1, idx_upper_bound_1]
            upper_bound_2 = quantiles[1, idx_upper_bound_2]
        else:
            upper_bound_1 = upper_bound_2 = prep.global_model_cfg[par + '_integral_upper_bound']

        return upper_bound_1, upper_bound_2

    def prior_convolutions(
        self, covs_1: np.ndarray, covs_2: np.ndarray,
        residuals_1: np.ndarray, residuals_2: np.ndarray,
        Rb_1: float, Rb_2: float,
        sig_Rb_1: float, sig_Rb_2: float,
        tau_Rb_1: float, tau_Rb_2: float,
        gamma_Rb_1: float, gamma_Rb_2: float,
        shift_Rb: float,
        Ebv_1: float, Ebv_2: float,
        tau_Ebv_1: float, tau_Ebv_2: float,
        gamma_Ebv_1: float, gamma_Ebv_2: float,
    ) -> tuple:
        
        upper_bound_Rb_1, upper_bound_Rb_2 = self.get_upper_bounds(
            prep.gRb_quantiles, gamma_Rb_1, gamma_Rb_2, "Rb"
        )
        upper_bound_Ebv_1, upper_bound_Ebv_2 = self.get_upper_bounds(
            prep.gEbv_quantiles, gamma_Ebv_1, gamma_Ebv_2, "Ebv"
        )

        norm_1 = sp_special.gammainc(gamma_Ebv_1, upper_bound_Ebv_1) * sp_special.gamma(gamma_Ebv_1)
        norm_2 = sp_special.gammainc(gamma_Ebv_2, upper_bound_Ebv_2) * sp_special.gamma(gamma_Ebv_2)

        if gamma_Rb_1 != NULL_VALUE:
            norm_1 *= sp_special.gammainc(gamma_Rb_1, upper_bound_Rb_1) * sp_special.gamma(gamma_Rb_1)
        if gamma_Rb_2 != NULL_VALUE:
            norm_2 *= sp_special.gammainc(gamma_Rb_2, upper_bound_Rb_2) * sp_special.gamma(gamma_Rb_2)

        probs, status = self.convolution_fn(
            cov_1=covs_1, cov_2=covs_2, res_1=residuals_1, res_2=residuals_2,
            Rb_1=Rb_1, Rb_2=Rb_2,
            sig_Rb_1=sig_Rb_1, sig_Rb_2=sig_Rb_2,
            tau_Rb_1=tau_Rb_1, tau_Rb_2=tau_Rb_2,
            gamma_Rb_1=gamma_Rb_1, gamma_Rb_2=gamma_Rb_2,
            Ebv_1=Ebv_1, Ebv_2=Ebv_2,
            tau_Ebv_1=tau_Ebv_1, tau_Ebv_2=tau_Ebv_2,
            gamma_Ebv_1=gamma_Ebv_1, gamma_Ebv_2=gamma_Ebv_2,
            lower_bound_Rb_1=prep.global_model_cfg.Rb_integral_lower_bound,
            lower_bound_Rb_2=prep.global_model_cfg.Rb_integral_lower_bound,
            upper_bound_Rb_1=upper_bound_Rb_1, upper_bound_Rb_2=upper_bound_Rb_2,
            lower_bound_Ebv_1=prep.global_model_cfg.Ebv_integral_lower_bound,
            lower_bound_Ebv_2=prep.global_model_cfg.Ebv_integral_lower_bound,
            upper_bound_Ebv_1=upper_bound_Ebv_1, upper_bound_Ebv_2=upper_bound_Ebv_2,
            shift_Rb=shift_Rb, selection_bias_correction=prep.selection_bias_correction,
        )

        p_1 = probs[:, 0] / norm_1
        p_2 = probs[:, 1] / norm_2

        p1_nans = np.isnan(p_1)
        p2_nans = np.isnan(p_2)

        if np.any(p1_nans):
            print("Pop 1 contains nan probabilities:", np.count_nonzero(p1_nans)/len(p1_nans)*100, "%")
            print("Pop 1 pars:", [Rb_1, sig_Rb_1, Ebv_1, gamma_Ebv_1])
            print("Pop 1 norm:", norm_1, "\n")
        if np.any(p2_nans):
            print("Pop 2 contains nan probabilities:", np.count_nonzero(p2_nans)/len(p2_nans)*100, "%")
            print("Pop 1 pars:", [Rb_2, sig_Rb_2, Ebv_2, gamma_Ebv_2])
            print("Pop 2 norm:", norm_2, "\n")

        return p_1, p_2, status

    def independent_mvgaussian(
        self, means: np.ndarray, sigmas: np.ndarray
    ):

        idx_not_observed = prep.host_galaxy_observables == NULL_VALUE
        n_properties = prep.host_galaxy_observables.shape[1]
        res = np.atleast_3d(
            np.where(
                idx_not_observed,
                0.,
                prep.host_galaxy_observables - means
            )
        )

        sigmas = np.tile(
            sigmas, [prep.host_galaxy_observables.shape[0], 1]
        )
        sigmas = np.where(
            idx_not_observed,
            0.,
            sigmas
        )
        sigmas = np.eye(n_properties) * sigmas[:, None, :]
        cov = prep.host_galaxy_covariances + sigmas**2

        exponent = np.squeeze(
            np.moveaxis(res, 1, 2) @ np.linalg.inv(cov) @ res
        )

        prob = (
            ((2 * np.pi)**(-n_properties/2) *
            np.linalg.det(cov)**(-1/2)) *
            np.exp(-0.5 * exponent)
        )

        return prob

    def host_galaxy_probs(
        self,
        host_galaxy_means: np.ndarray,
        host_galaxy_sigmas: np.ndarray,
    ):

        mu_1, mu_2 = host_galaxy_means[::2], host_galaxy_means[1::2]
        sig_1, sig_2 = host_galaxy_sigmas[::2], host_galaxy_sigmas[1::2]

        prob_1 = self.independent_mvgaussian(mu_1, sig_1)
        prob_2 = self.independent_mvgaussian(mu_2, sig_2)

        return prob_1, prob_2

    def volumetric_sn_rates(
        self, observed_redshifts: np.ndarray,
        cosmo: apy_cosmo.Cosmology,
        eta: float, prompt_fraction: float,
    ):

        dtd_t0 = prep.global_model_cfg['dtd_cfg']['t0']
        dtd_t1 = prep.global_model_cfg['dtd_cfg']['t1']

        H0 = cosmo.H0.value
        H0_gyrm1 = cosmo.H0.to(1/Gyr).value
        Om0 = cosmo.Om0
        w0 = cosmo.w0
        wa = cosmo.wa
        cosmo_args = (H0_gyrm1,Om0,1.-Om0,w0, wa)

        convolution_time_limits = cosmo_utils.convolution_limits(
            cosmo, observed_redshifts, dtd_t0, dtd_t1
        )
        minimum_convolution_time = np.min(convolution_time_limits)
        
        warnings.filterwarnings("error")
        try:
            z0 = apy_cosmo.z_at_value(
                cosmo.age, minimum_convolution_time * Gyr,
                method='Bounded'
            )
        except:
            return np.ones_like(observed_redshifts) * -np.inf
        warnings.resetwarnings()
        zinf = 20.
        age_of_universe = cosmo.age(0).value - cosmo.age(zinf).value
        _, zs, _ = cosmo_utils.redshift_at_times(
            convolution_time_limits, minimum_convolution_time, z0, cosmo_args
        )
        integral_limits = np.array(zs.tolist(), dtype=np.float64)
        sn_rates = cosmo_utils.volumetric_rates(
            observed_redshifts, integral_limits, H0, Om0,
            w0, wa, eta, prompt_fraction, zinf=zinf,
            age=age_of_universe
        )

        return sn_rates

    def volumetric_rate_probs(
        self, cosmo: apy_cosmo.Cosmology,
        eta: float, prompt_fraction: float,
    ):
        
        sn_rates = self.volumetric_sn_rates(
            prep.observed_volumetric_rate_redshifts, cosmo,
            eta, prompt_fraction
        )

        is_inf = np.any(np.isinf(sn_rates))
        if is_inf:
            return -np.inf

        obs_volumetric_rates = prep.observed_volumetric_rates
        obs_volumetric_rate_errors = prep.observed_volumetric_rate_errors

        normalization = -0.5 * np.log(2 * np.pi) + np.log(obs_volumetric_rate_errors)
        exponent = -0.5 * (obs_volumetric_rates - sn_rates[:,0])**2 / obs_volumetric_rate_errors**2
        log_probs = normalization + exponent
        log_prob = np.sum(log_probs)

        return log_prob

    def reduce_duplicates(
        self, probs: np.ndarray,
    ) -> tuple:
        
        unique_sn_probs, duplicate_sn_probs = prep.reorder_duplicates(
            probs, prep.idx_unique_sn, prep.idx_duplicate_sn
        )
        reduced_duplicate_sn_probs = np.array(
            [
                np.prod(duplicate_probs) for duplicate_probs in duplicate_sn_probs
            ]
        )
        reduced_dulpicate_sn_log_probs = np.array(
            [np.sum(np.log(duplicate_probs)) for duplicate_probs in duplicate_sn_probs]
        )
        probs = np.concatenate([unique_sn_probs, reduced_duplicate_sn_probs])
        log_probs = np.concatenate([np.log(unique_sn_probs), reduced_dulpicate_sn_log_probs])

        return probs, log_probs

    def log_likelihood(
        self,
        Mb_1: float, Mb_2: float,
        s_1: float, s_2: float,
        c_1: float, c_2: float,
        alpha_1: float, alpha_2: float,
        beta_1: float, beta_2: float,
        sig_int_1: float, sig_int_2: float,
        sig_s_1: float, sig_s_2: float,
        sig_c_1: float, sig_c_2: float,
        Rb_1: float, Rb_2: float,
        sig_Rb_1: float, sig_Rb_2: float,
        tau_Rb_1: float, tau_Rb_2: float,
        gamma_Rb_1: float, gamma_Rb_2: float,
        shift_Rb: float,
        Ebv_1: float, Ebv_2: float,
        tau_Ebv_1: float, tau_Ebv_2: float,
        gamma_Ebv_1: float, gamma_Ebv_2: float,
        host_galaxy_means: np.ndarray,
        host_galaxy_sigs: np.ndarray,
        w: float, H0: float, Om0: float,
        w0: float, wa: float, eta: float,
        prompt_fraction: float
    ) -> float:
        
        cosmo = apy_cosmo.Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)

        sn_cov_1, sn_cov_2 = self.population_covariances(
            alpha_1=alpha_1, alpha_2=alpha_2,
            beta_1=beta_1, beta_2=beta_2,
            sig_int_1=sig_int_1, sig_int_2=sig_int_2,
            sig_s_1=sig_s_1, sig_s_2=sig_s_2,
            sig_c_1=sig_c_1, sig_c_2=sig_c_2,
        )
        sn_residuals_1, sn_residuals_2 = self.population_residuals(
            Mb_1=Mb_1, Mb_2=Mb_2,
            s_1=s_1, s_2=s_2,
            c_1=c_1, c_2=c_2,
            alpha_1=alpha_1, alpha_2=alpha_2,
            beta_1=beta_1, beta_2=beta_2,
            cosmo=cosmo
        )

        use_gaussian_Rb = (
            gamma_Rb_1 == NULL_VALUE and
            gamma_Rb_2 == NULL_VALUE and
            sig_Rb_1 != NULL_VALUE and
            sig_Rb_2 != NULL_VALUE
        )

        if use_gaussian_Rb:
            self.convolution_fn = _Ebv_prior_convolution
        else:
            self.convolution_fn = _Ebv_Rb_prior_convolution
        
        sn_probs_1, sn_probs_2, status = self.prior_convolutions(
            covs_1=sn_cov_1, covs_2=sn_cov_2,
            residuals_1=sn_residuals_1, residuals_2=sn_residuals_2,
            Rb_1=Rb_1, Rb_2=Rb_2,
            sig_Rb_1=sig_Rb_1, sig_Rb_2=sig_Rb_2,
            tau_Rb_1=tau_Rb_1, tau_Rb_2=tau_Rb_2,
            gamma_Rb_1=gamma_Rb_1, gamma_Rb_2=gamma_Rb_2,
            shift_Rb=shift_Rb,
            Ebv_1=Ebv_1, Ebv_2=Ebv_2,
            tau_Ebv_1=tau_Ebv_1, tau_Ebv_2=tau_Ebv_2,
            gamma_Ebv_1=gamma_Ebv_1, gamma_Ebv_2=gamma_Ebv_2,
        )
        sn_probs_1, sn_log_probs_1 = self.reduce_duplicates(sn_probs_1)
        sn_probs_2, sn_log_probs_2 = self.reduce_duplicates(sn_probs_2)
        reduced_status = (
            self.reduce_duplicates(status[:,0])[0] * self.reduce_duplicates(status[:,1])[0]
        ).astype('bool')

        idx_prior_convolution_failed = ~reduced_status
        idx_sn_prob_below_zero = (sn_probs_1 < 0.) | (sn_probs_2 < 0.)
        idx_prob_not_finite = (
            ~np.isfinite(sn_probs_1) | ~np.isfinite(sn_probs_2)
        )
        idx_not_valid = (
            idx_prior_convolution_failed |
            idx_sn_prob_below_zero |
            idx_prob_not_finite
        )
        sn_prob_not_valid = np.any(idx_not_valid)
        if sn_prob_not_valid:
            
            n_convolution_failed = np.sum(idx_prior_convolution_failed)
            n_sn_prob_below_zero = np.sum(idx_sn_prob_below_zero)
            n_prob_not_finite = np.sum(idx_prob_not_finite)

            warning_string = (
                "\n --------- Failure in SN prior convolution --------- \n" +
                f"\nNo. of SN with failed convolutions: {n_convolution_failed}\n" +
                f"No. of SN with negative probabilities: {n_sn_prob_below_zero}\n" +
                f"No. of SN with non-finite probabilities: {n_prob_not_finite}\n"
            )
            failure_in_pop_1 = (
                np.any(~status[:,0]) or
                np.any(sn_probs_1 < 0.) or
                np.any(~np.isfinite(sn_probs_1))
            )
            failure_in_pop_2 = (
                np.any(~status[:,1]) or
                np.any(sn_probs_2 < 0.) or
                np.any(~np.isfinite(sn_probs_2))
            )

            if failure_in_pop_1:
                warning_string += (
                    "\nFailure(s) occured in population 1.\n" +
                    f"Mb: {Mb_1:.3f}, x1: {s_1:.3f}, c: {c_1:.3f},\n" +
                    f"alpha: {alpha_1:.3f}, beta: {beta_1:.3f},\n" +
                    f"sig_int: {sig_int_1:.3f}, sig_s: {sig_s_1:.3f}, sig_c: {sig_c_1:.3f},\n" +
                    f"Rb: {Rb_1:.3f}, sig_Rb: {sig_Rb_1:.3f},\n" +
                    f"tau_Rb: {tau_Rb_1:.3f}, gamma_Rb: {gamma_Rb_1:.3f}, shift_Rb: {shift_Rb:.3f},\n" +
                    f"Ebv: {Ebv_1:.3f}, tau_Ebv: {tau_Ebv_1:.3f}, gamma_Ebv: {gamma_Ebv_1:.3f}\n"
                )
            
            if failure_in_pop_2:
                warning_string += (
                    "\nFailure(s) occured in population 2.\n" +
                    f"Mb: {Mb_2:.3f}, x1: {s_2:.3f}, c: {c_2:.3f},\n" +
                    f"alpha: {alpha_2:.3f}, beta: {beta_2:.3f},\n" +
                    f"sig_int: {sig_int_2:.3f}, sig_s: {sig_s_2:.3f}, sig_c: {sig_c_2:.3f},\n" +
                    f"Rb: {Rb_2:.3f}, sig_Rb: {sig_Rb_2:.3f},\n" +
                    f"tau_Rb: {tau_Rb_2:.3f}, gamma_Rb: {gamma_Rb_2:.3f}, shift_Rb: {shift_Rb:.3f},\n" +
                    f"Ebv: {Ebv_2:.3f}, tau_Ebv: {tau_Ebv_2:.3f}, gamma_Ebv: {gamma_Ebv_2:.3f}\n"
                )

            warning_string += (
                "\nCosmology parameters used:\n" +
                f"H0: {H0:.3f}, Om0: {Om0:.3f}, w0: {w0:.3f}, wa: {wa}\n" +
                f"eta: {eta:.3f}, prompt_fraction: {prompt_fraction:.3f}\n"
            )

            cid_for_failures = np.concatenate(
                prep.reorder_duplicates(
                    prep.sn_cids, prep.idx_unique_sn,
                    prep.idx_duplicate_sn
                )
            )[idx_not_valid]
            calibrator_flag_for_failures = prep.idx_reordered_calibrator_sn[idx_not_valid]
            observed_mb_for_failures = np.concatenate(
                prep.reorder_duplicates(
                    prep.sn_observables[:,0], prep.idx_unique_sn,
                    prep.idx_duplicate_sn
                )
            )[idx_not_valid]
            observed_x1_for_failures = np.concatenate(
                prep.reorder_duplicates(
                    prep.sn_observables[:,1], prep.idx_unique_sn,
                    prep.idx_duplicate_sn
                )
            )[idx_not_valid]
            observed_c_for_failures = np.concatenate(
                prep.reorder_duplicates(
                    prep.sn_observables[:,2], prep.idx_unique_sn,
                    prep.idx_duplicate_sn
                )
            )[idx_not_valid]
            observed_redshifts_for_failures = np.concatenate(
                prep.reorder_duplicates(
                    prep.sn_observables[:,-1], prep.idx_unique_sn,
                    prep.idx_duplicate_sn
                )
            )[idx_not_valid]
            
            warning_string += "\nObserved SN parameters:\n"
            for cid, calibrator_flag, mb, x1, c, z in zip(
                cid_for_failures, calibrator_flag_for_failures,
                observed_mb_for_failures, observed_x1_for_failures,
                observed_c_for_failures, observed_redshifts_for_failures
            ):
                warning_string += (
                    f"\ncid: {cid}\n" +
                    f"calibrator: {calibrator_flag}\n" +
                    f"mb: {mb:.3f}\n" +
                    f"x1: {x1:.3f}\n" +
                    f"c: {c:.3f}\n" +
                    f"redshift: {z:.3f}\n"
                )

            warning_string += "\n ----------------------------------------------- \n"

            warnings.warn(warning_string)

            return (
                -np.inf,
                np.ones(prep.n_unique_sn)*np.nan,
                np.ones(prep.n_unique_sn)*np.nan,
                np.ones(prep.n_unique_sn)*np.nan,
            )
        
        use_physical_population_fraction = prep.global_model_cfg["use_physical_ratio"]

        if use_physical_population_fraction:
            # TODO: FIX FOR DUPLICATE SN
            sn_redshifts = prep.sn_observables[:,-1]
            sn_rates = self.volumetric_sn_rates(
                observed_redshifts=sn_redshifts,
                cosmo=cosmo, eta=eta,
                prompt_fraction=prompt_fraction
            )

            is_inf = np.any(np.isinf(sn_rates))
            if is_inf:
                return (
                    -np.inf,
                    np.ones(prep.n_unique_sn)*np.nan,
                    np.ones(prep.n_unique_sn)*np.nan,
                    np.ones(prep.n_unique_sn)*np.nan,
                )

            w_vector = sn_rates[:, -1] / sn_rates[:, 0]
        else:
            w_vector = np.ones_like(sn_probs_1) * w
        log_w_1 = np.log(w)
        log_w_2 = np.log(1-w)

        if host_galaxy_means.shape[0] > 0:
            host_probs_1, host_probs_2 = self.host_galaxy_probs(
                host_galaxy_means=host_galaxy_means,
                host_galaxy_sigmas=host_galaxy_sigs,
            )
            host_probs_1, host_log_probs_1 = self.reduce_duplicates(host_probs_1)
            host_probs_2, host_log_probs_2 = self.reduce_duplicates(host_probs_2)
        else:
            host_probs_1, host_log_probs_1 = np.ones(prep.n_unique_sn), np.zeros(prep.n_unique_sn)
            host_probs_2, host_log_probs_2 = np.ones(prep.n_unique_sn), np.zeros(prep.n_unique_sn)

        pop_1_log_probs = log_w_1 + sn_log_probs_1 + host_log_probs_1
        pop_2_log_probs = log_w_2 + sn_log_probs_2 + host_log_probs_2
        combined_log_probs = np.logaddexp(pop_1_log_probs, pop_2_log_probs)
        
        if prep.global_model_cfg['only_evaluate_calibrators']:
            combined_log_probs = combined_log_probs[~prep.idx_reordered_calibrator_sn]
        
        log_host_membership_probs = (
            1./np.log(10) * (
                log_w_1 + host_log_probs_1 - log_w_2 - host_log_probs_2
            ).flatten()
        )
        log_sn_membership_probs = (
            1./np.log(10) * (
                log_w_1 + sn_log_probs_1 - log_w_2 - sn_log_probs_2
            ).flatten()
        )
        log_full_membership_probs = (
            1./np.log(10) * (pop_1_log_probs - pop_2_log_probs).flatten()
        )

        use_volumetric_rates = prep.global_model_cfg["use_volumetric_rates"]
        if use_volumetric_rates and use_physical_population_fraction:
            log_volumetric_prob = self.volumetric_rate_probs(
                cosmo, eta=eta, prompt_fraction=prompt_fraction
            )
            log_prob = np.sum(combined_log_probs) + log_volumetric_prob
        else:
            log_prob = np.sum(combined_log_probs)

        if not np.isfinite(log_prob):
            warnings.warn(
                "Log posterior probability is not finite. Returning -np.inf. "
            )
            log_prob = -np.inf
            log_full_membership_probs = np.ones(prep.n_unique_sn)*np.nan
            log_host_membership_probs = np.ones(prep.n_unique_sn)*np.nan
            log_sn_membership_probs = np.ones(prep.n_unique_sn)*np.nan
        
        self.convolution_fn = None

        return log_prob, log_full_membership_probs, log_sn_membership_probs, log_host_membership_probs

    def __call__(self, theta: np.ndarray) -> float:

        n_host_galaxy_observables = (
            prep.host_galaxy_observables.shape[1] - prep.n_unused_host_properties
        )
        param_dict = utils.theta_to_dict(
            theta=theta, shared_par_names=prep.global_model_cfg.shared_par_names,
            independent_par_names=prep.global_model_cfg.independent_par_names,
            n_host_galaxy_observables=n_host_galaxy_observables,
            n_unused_host_properties=prep.n_unused_host_properties,
            ratio_par_name=prep.global_model_cfg.ratio_par_name,
            cosmology_par_names=prep.global_model_cfg.cosmology_par_names,
            use_physical_ratio=prep.global_model_cfg.use_physical_ratio,
        )

        log_prior = self.log_prior(param_dict)
        if np.isinf(log_prior):
            return (
                log_prior,
                np.ones(prep.n_unique_sn)*np.nan,
                np.ones(prep.n_unique_sn)*np.nan,
                np.ones(prep.n_unique_sn)*np.nan,
            )
        
        # TODO: Update to handle cosmology
        if prep.global_model_cfg.use_sigmoid:
            param_dict = utils.apply_sigmoid(
                param_dict, sigmoid_cfg=prep.global_model_cfg.sigmoid_cfg,
                independent_par_names=prep.global_model_cfg.independent_par_names,
                ratio_par_name=prep.global_model_cfg.ratio_par_name
            )

        preset_values = prep.global_model_cfg.preset_values
        for par in preset_values.keys():
            current_value = param_dict.get(par, None)
            is_null = current_value == NULL_VALUE
            is_none = current_value is None
            update_par = is_null or is_none
            if update_par:
                param_dict[par] = preset_values[par]

        (
            log_likelihood, log_full_membership_probs,
            log_sn_membership_probs, log_host_membership_probs
        ) = self.log_likelihood(**param_dict)

        return (
            log_likelihood, log_full_membership_probs,
            log_sn_membership_probs, log_host_membership_probs
        )

class TrippModel():

    def __init__(
        self, model_cfg: dict
    ):
        self.cfg = model_cfg
        pass
    
    def logprior(self, par_dict: dict) -> float:
            
        log_prior = 0.0
        for par in par_dict.keys():
            par_value = par_dict[par]
            is_in_priors = par in self.cfg["prior_bounds"].keys()
            if is_in_priors:
                log_prior += utils.uniform(
                    par_value, **self.cfg["prior_bounds"][par]
                )
            if np.isinf(log_prior):
                break
        
        return log_prior

    def input_to_dict(self, theta: np.ndarray) -> dict:
        
        pars_to_fit = self.cfg["pars"]
        input_dict = {par: value for par, value in zip(pars_to_fit, theta)}
        input_dict = {**self.cfg["preset_values"], **input_dict}

        return input_dict

    def residuals(
        self, observables: np.ndarray,
        Mb: float, alpha: float, beta: float,
        cosmo: apy_cosmo.Cosmology
    ):
        
        mb = observables[:, 0]
        x1 = observables[:, 1]
        c = observables[:, 2]
        z = observables[:, 3]

        distance_moduli = cosmo.distmod(z).value
        residuals = mb - Mb - distance_moduli + alpha * x1 - beta * c

        return residuals
    
    def variance(
        self, observables: np.ndarray,
        covariances: np.ndarray, 
        alpha: float, beta: float, sig_int: float
    ):
        
        disp_v_pec = 200. # km / s
        c = 300000. # km / s

        z = observables[:, 3]
        covs = covariances
        if len(covs.shape) == 2:
            covs = np.tile(covs, [len(z), 1, 1])

        var_tmp = np.diagonal(covs, axis1=1, axis2=2)
        var = var_tmp.copy()
        var[:,1] = var_tmp[:,1] * alpha**2
        var[:,2] = var_tmp[:,2] * beta**2
        var = np.sum(var, axis=1)
        var += (5 / np.log(10))**2 * (disp_v_pec / (c * z))**2
        var -= 2 * beta * covs[:, 0, 2]
        var += 2 * alpha * covs[:, 0, 1]
        var -= 2 * alpha * beta * covs[:, 1, 2]
        var += sig_int**2

        return var

    def log_likelihood(
        self, observables: np.ndarray,
        covariances: np.ndarray, 
        Mb: float, alpha: float, beta: float, sig_int: float,
        H0: float, Om0: float, w0: float, wa: float
    ):
        
        cosmo = apy_cosmo.Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
        residuals = self.residuals(observables, Mb, alpha, beta, cosmo)
        var = self.variance(observables, covariances, alpha, beta, sig_int)

        if np.any(var <= 0):
            return -np.inf

        log_likelihood = -0.5 * np.sum(residuals**2 / var + np.log(var))

        return log_likelihood

    def __call__(
        self, theta: np.ndarray, observables: np.ndarray, covariances: np.ndarray
    ) -> float:

        par_dict = self.input_to_dict(theta)
        log_prior = self.logprior(par_dict)
        if np.isinf(log_prior):
            return -np.inf
        log_likelihood = self.log_likelihood(observables, covariances, **par_dict)

        return log_likelihood