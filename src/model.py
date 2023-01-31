import numpy as np
import numba as nb
import src.utils as utils
import scipy.stats as stats
import scipy.special as sp_special

from NumbaQuadpack import quadpack_sig, dqags
from astropy.cosmology import Planck18 as cosmo

NULL_VALUE = -9999.0

# ---------- E(B-V) PRIOR INTEGRAL ------------

@nb.jit
def Ebv_integral_body(
    x, i1, i2, i3, i5,
    i6, i9, r1, r2, r3,
    rb, sig_rb, Ebv, tau_Ebv, gamma_Ebv
):  

    if tau_Ebv == NULL_VALUE:
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
def Ebv_integral(x, data):
    _data = nb.carray(data, (17,))
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
    tau_Ebv = _data[15]
    gamma_Ebv = _data[16]
    return Ebv_integral_body(
        x, i1, i2, i3, i5, i6, i9, r1, r2, r3, 
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
    shift_Rb: float
):

    n_sn = len(cov_1)
    probs = np.zeros((n_sn, 2))
    status = np.ones((n_sn, 2), dtype='bool')
    params_1 = np.array([Rb_1, sig_Rb_1, Ebv_1, tau_Ebv_1, gamma_Ebv_1])
    params_2 = np.array([Rb_2, sig_Rb_2, Ebv_2, tau_Ebv_2, gamma_Ebv_2])

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

    inner_value, _, _ = dqags(
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

    def __init__(
        self, cfg: dict, observables: np.ndarray, covariances: np.ndarray
    ) -> None:

        self.mb = observables[:, 0]
        self.s = observables[:, 1]
        self.c = observables[:, 2]
        self.z = observables[:, 3]
        self.covs = covariances

        self.shared_par_names = cfg['shared_par_names']
        self.independent_par_names = cfg['independent_par_names']
        self.stretch_par_name = cfg['stretch_par_name']
        self.ratio_par_name = cfg['ratio_par_name']
        self.use_sigmoid = cfg['use_sigmoid']
        self.sigmoid_cfg = cfg['sigmoid_cfg']
        self.prior_bounds = cfg['prior_bounds']
        self.Ebv_integral_lower_bound = cfg.get('Ebv_integral_lower_bound', 0)
        self.Ebv_integral_upper_bound = cfg.get('Ebv_integral_upper_bound', -9999.)
        self.Rb_integral_lower_bound = cfg.get('Rb_integral_lower_bound', 0)
        self.Rb_integral_upper_bound = cfg.get('Rb_integral_upper_bound', -9999.)

        if self.stretch_par_name in self.independent_par_names:
            self.stretch_independent = True
        else:
            self.stretch_independent = False

        self.gEbv_quantiles = self.set_gamma_quantiles(cfg, 'Ebv')
        self.gRb_quantiles = self.set_gamma_quantiles(cfg, 'Rb')

        self.Ebv_prior_conv_fn = _Ebv_prior_convolution
        self.Rb_Ebv_prior_conv_fn = _Ebv_Rb_prior_convolution

        self.__dict__.update(cfg['preset_values'])
    
    def set_gamma_quantiles(self, cfg: dict, par: str) -> np.ndarray:

        if self.__dict__[par + "_integral_upper_bound"] == NULL_VALUE:
            quantiles = utils.create_gamma_quantiles(
                cfg['prior_bounds']['gamma_' + par]['lower'],
                cfg['prior_bounds']['gamma_' + par]['upper'],
                cfg['resolution_g' + par], cfg['cdf_limit_g' + par]
            )
        else:
            quantiles = None

        return quantiles

    def log_prior(self, par_dict: dict) -> float:
        
        value = 0.
        stretch_1_par = self.stretch_par_name + "_1"
        stretch_2_par = self.stretch_par_name + "_2"

        for value_key in par_dict.keys():
            
            if par_dict[value_key] == NULL_VALUE:
                continue

            # TODO: Remove 3-deep conditionals bleeeeh
            bounds_key = ""
            is_independent_stretch = (
                (
                    value_key == stretch_1_par or value_key == stretch_2_par
                ) and self.stretch_independent
            )
            is_not_ratio_par_name = value_key != self.ratio_par_name

            if is_independent_stretch:
                if not par_dict[stretch_1_par] < par_dict[stretch_2_par]:
                    value += -np.inf
                    break
            
            if is_not_ratio_par_name:
                bounds_key = "_".join(value_key.split("_")[:-1])
            else:
                bounds_key = value_key
            
            is_in_priors = bounds_key in self.prior_bounds.keys()
            if is_in_priors:
                value += utils.uniform(
                    par_dict[value_key], **self.prior_bounds[bounds_key]
                )

            if np.isinf(value):
                break

        return value
    
    def population_covariances(self) -> np.ndarray:

        cov = np.tile(self.covs, (2, 1, 1, 1))
        disp_v_pec = 200. # km / s
        c = 300000. # km / s
        
        cov[:,:,0,0] += np.array([
            [self.sig_int_1**2 + self.alpha_1**2 * self.sig_s_1**2 + self.beta_1**2 * self.sig_c_1**2],
            [self.sig_int_2**2 + self.alpha_2**2 * self.sig_s_2**2 + self.beta_2**2 * self.sig_c_2**2]
        ])
        cov[:,:,0,0] += np.tile(
            self.z**(-2) * (
                (5. / np.log(10.))**2
                * (disp_v_pec / c)**2
            ), (2, 1)
        )

        cov[:,:,1,1] += np.array([
            [self.sig_s_1**2], [self.sig_s_2**2]
        ])
        cov[:,:,2,2] += np.array([
            [self.sig_c_1**2], [self.sig_c_2**2]
        ])
        cov[:,:,0,1] += np.array([
            [self.alpha_1 * self.sig_s_1**2], [self.alpha_2 * self.sig_s_2**2]
        ])
        cov[:,:,0,2] += np.array([
            [self.beta_1 * self.sig_c_1**2], [self.beta_2 * self.sig_c_2**2]
        ])
        cov[:,:,1,0] = cov[:,:,0,1]
        cov[:,:,2,0] = cov[:,:,0,2]

        return cov[0], cov[1]

    def population_residuals(self) -> np.ndarray:
        
        residuals = np.zeros((2, len(self.mb), 3))
        distance_moduli = np.tile(
            cosmo.distmod(self.z).value, (2, 1)
        ) + np.log10(cosmo.H0.value / self.H0)

        residuals[:, :, 0] = np.tile(self.mb, (2, 1)) - np.array([
            [self.Mb_1 + self.alpha_1 * self.s_1 + self.beta_1 * self.c_1],
            [self.Mb_2 + self.alpha_2 * self.s_2 + self.beta_2 * self.c_2]
        ]) - distance_moduli
        residuals[:, :, 1] = np.tile(self.s, (2, 1)) - np.array([
            [self.s_1], [self.s_2]
        ])
        residuals[:, :, 2] = np.tile(self.c, (2, 1)) - np.array([
            [self.c_1], [self.c_2]
        ])

        return residuals[0], residuals[1]
    
    def get_upper_bounds(self, par: str) -> tuple[float]:

        var_dict = self.__dict__
        
        if (
            var_dict["g" + par + "_quantiles"] is not None and
            var_dict['gamma_' + par + '_1'] != NULL_VALUE and
            var_dict['gamma_' + par + '_2'] != NULL_VALUE
        ):
            quantiles = var_dict["g" + par + "_quantiles"]
            idx_upper_bound_1 = utils.find_nearest_idx(quantiles[0], var_dict['gamma_' + par + '_1'])
            idx_upper_bound_2 = utils.find_nearest_idx(quantiles[0], var_dict['gamma_' + par + '_2'])
            upper_bound_1 = quantiles[1, idx_upper_bound_1]
            upper_bound_2 = quantiles[1, idx_upper_bound_2]
        else:
            upper_bound_1 = upper_bound_2 = var_dict[par + '_integral_upper_bound']

        return upper_bound_1, upper_bound_2

    def prior_convolutions(
        self, covs_1: np.ndarray, covs_2: np.ndarray,
        residuals_1: np.ndarray, residuals_2: np.ndarray
    ) -> float:
        
        upper_bound_Rb_1, upper_bound_Rb_2 = self.get_upper_bounds('Rb')
        upper_bound_Ebv_1, upper_bound_Ebv_2 = self.get_upper_bounds('Ebv')

        norm_1 = sp_special.gammainc(self.gamma_Ebv_1, upper_bound_Ebv_1) * sp_special.gamma(self.gamma_Ebv_1)
        norm_2 = sp_special.gammainc(self.gamma_Ebv_2, upper_bound_Ebv_2) * sp_special.gamma(self.gamma_Ebv_2)

        if self.gamma_Rb_1 != NULL_VALUE:
            norm_1 *= sp_special.gammainc(self.gamma_Rb_1, upper_bound_Rb_1) * sp_special.gamma(self.gamma_Rb_1)
        if self.gamma_Rb_2 != NULL_VALUE:
            norm_2 *= sp_special.gammainc(self.gamma_Rb_2, upper_bound_Rb_2) * sp_special.gamma(self.gamma_Rb_2)

        probs, status = self.convolution_fn(
            cov_1=covs_1, cov_2=covs_2, res_1=residuals_1, res_2=residuals_2,
            Rb_1=self.Rb_1, Rb_2=self.Rb_2,
            sig_Rb_1=self.sig_Rb_1, sig_Rb_2=self.sig_Rb_2,
            tau_Rb_1=self.tau_Rb_1, tau_Rb_2=self.tau_Rb_2,
            gamma_Rb_1=self.gamma_Rb_1, gamma_Rb_2=self.gamma_Rb_2,
            Ebv_1=self.Ebv_1, Ebv_2=self.Ebv_2,
            tau_Ebv_1=self.tau_Ebv_1, tau_Ebv_2=self.tau_Ebv_2,
            gamma_Ebv_1=self.gamma_Ebv_1, gamma_Ebv_2=self.gamma_Ebv_2,
            lower_bound_Rb_1=self.Rb_integral_lower_bound, lower_bound_Rb_2=self.Rb_integral_lower_bound,
            upper_bound_Rb_1=upper_bound_Rb_1, upper_bound_Rb_2=upper_bound_Rb_2,
            lower_bound_Ebv_1=self.Ebv_integral_lower_bound, lower_bound_Ebv_2=self.Ebv_integral_lower_bound,
            upper_bound_Ebv_1=upper_bound_Ebv_1, upper_bound_Ebv_2=upper_bound_Ebv_2,
            shift_Rb=self.shift_Rb
        )

        p_1 = probs[:, 0] / norm_1
        p_2 = probs[:, 1] / norm_2

        p1_nans = np.isnan(p_1)
        p2_nans = np.isnan(p_2)

        if np.any(p1_nans):
            print("Pop 1 contains nan probabilities:", np.count_nonzero(p1_nans)/len(p1_nans)*100, "%")
            print("Pop 1 pars:", [self.Rb_1, self.sig_Rb_1, self.Ebv_1, self.gamma_Ebv_1])
            print("Pop 1 norm:", norm_1, "\n")
        if np.any(p2_nans):
            print("Pop 2 contains nan probabilities:", np.count_nonzero(p2_nans)/len(p2_nans)*100, "%")
            print("Pop 1 pars:", [self.Rb_2, self.sig_Rb_2, self.Ebv_2, self.gamma_Ebv_2])
            print("Pop 2 norm:", norm_2, "\n")

        return p_1, p_2, status

    def log_likelihood(self) -> float:
        
        cov_1, cov_2 = self.population_covariances()
        residuals_1, residuals_2 = self.population_residuals()

        use_gaussian_Rb = (
            self.gamma_Rb_1 == NULL_VALUE and
            self.gamma_Rb_2 == NULL_VALUE and
            self.sig_Rb_1 != NULL_VALUE and
            self.sig_Rb_2 != NULL_VALUE
        )

        if use_gaussian_Rb:
            self.convolution_fn = self.Ebv_prior_conv_fn
        else:
            self.convolution_fn = self.Rb_Ebv_prior_conv_fn
        
        probs_1, probs_2, status = self.prior_convolutions(
            covs_1=cov_1, covs_2=cov_2,
            residuals_1=residuals_1, residuals_2=residuals_2
        )

        if np.any(probs_1 < 0.) | np.any(probs_2 < 0.):
            print("\nOh no, someones below 0\n")
            return -np.inf

        # TODO: Fix numerical stability by using logsumexp somehow
        log_prob = np.sum(
            np.log(
                self.w * probs_1 + (1-self.w) * probs_2
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

    def __call__(self, theta: np.ndarray) -> float:

        param_dict = utils.theta_to_dict(
            theta=theta, shared_par_names=self.shared_par_names,
            independent_par_names=self.independent_par_names,
            ratio_par_name=self.ratio_par_name
        )

        log_prior = self.log_prior(param_dict)
        if np.isinf(log_prior):
            return log_prior

        self.__dict__.update(param_dict)
        
        if self.use_sigmoid:
            self.__dict__.update(
                utils.apply_sigmoid(
                    self.__dict__, sigmoid_cfg=self.sigmoid_cfg,
                    independent_par_names=self.independent_par_names,
                    ratio_par_name=self.ratio_par_name
                )
            )
        
        log_likelihood = self.log_likelihood()

        return log_likelihood
    