import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import numba as nb
import jax.numpy as jnp
import src.utils as utils
import scipy.stats as stats
import scipy.special as sp_special
import astropy.cosmology as apy_cosmo

from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

import src.preprocessing as prep

NULL_VALUE = -9999.0
H0_CONVERSION_FACTOR = 0.001022
DH_70 = 4282.7494

# ---------- E(B-V) PRIOR INTEGRAL ------------

@nb.jit
def Ebv_integral_body(
    x, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, r1, r2, r3,
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
    _data = nb.carray(data, (17,))
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
    rb = _data[12]
    sig_rb = _data[13]
    Ebv = _data[14]
    tau_Ebv = _data[15]
    gamma_Ebv = _data[16]
    return Ebv_integral_body(
        x, i1, i2, i3, i4, i5, i6, i7, i8, i9, r1, r2, r3, 
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

# ---------- z(T) ODE ------------

def E(z, args):
    _, Om, Ode, w0, wa = args
    zp1 = 1+z
    #Ode = 1. - Om
    mass_term = Om * zp1**3.
    de_term = Ode * zp1**(3.*(1.+w0+wa)) * jnp.exp(-3. * wa * z/zp1)
    Ez = jnp.sqrt(mass_term + de_term)

    return Ez

def convolution_limits(cosmo, z, T0, T1):
    times_z0_lower = cosmo.age(z).value - T0
    times_z0_upper = cosmo.age(z).value - T1

    return jnp.concatenate((times_z0_lower, times_z0_upper))

def ode(
    t, z, args
):
    H0, _, _, _, _ = args
    zp1 = 1+z
    Ez = E(z, args)

    return -H0 * zp1 * Ez

@jax.jit
def redshift_at_times(
    conv_limit_times,
    t0, z0, cosmo_args
):

    idx_sort = jnp.argsort(conv_limit_times)
    idx_unsort = jnp.argsort(idx_sort)
    sorted_conv_limit_times = conv_limit_times[idx_sort]
    t1 = sorted_conv_limit_times[-1]

    term = ODETerm(ode)
    solver = Tsit5()
    saveat = SaveAt(ts=sorted_conv_limit_times)
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)

    sol = diffeqsolve(
        term, solver, t0=t0, t1=t1, y0=z0, dt0=None,
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=cosmo_args
    )

    ts = jnp.column_stack(
        jnp.split(
            sol.ts[idx_unsort], 2
        )
    )
    zs = jnp.column_stack(
        jnp.split(
            sol.ys[idx_unsort], 2
        )
    )
    num_steps = sol.stats['num_steps']

    return ts, zs, num_steps

# ---------- mu(z) via ODE ------------

def dc_ode(
    z, D, args
):
    Ez = E(z, args)

    return  1. / Ez

@jax.jit
def distance_modulus_at_redshift(
    z, cosmo_args
):
    z0 = 0.
    dc0 = 0.
    idx_sort = jnp.argsort(z)
    idx_unsort = jnp.argsort(idx_sort)
    sorted_z = z[idx_sort]
    z1 = sorted_z[-1]

    term = ODETerm(dc_ode)
    solver = Tsit5()
    saveat = SaveAt(ts=sorted_z)
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)

    sol = diffeqsolve(
        term, solver, t0=z0, t1=z1, y0=dc0, dt0=None,
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=cosmo_args
    )

    H0, _, _, _, _ = cosmo_args
    DH = DH_70 * 70./H0
    zs = sol.ts[idx_unsort]
    dcs = sol.ys[idx_unsort] * DH
    dls = (1+z) * dcs
    mus = 5.0 * jnp.log10(dls) + 25.0
    
    num_steps = sol.stats['num_steps']

    return mus

# ---------- N(z) integral ------------

@nb.jit
def lookback_time_integrand(
    z, args
):

    H0, Om, w0, wa = args
    H0_reduced = H0 * H0_CONVERSION_FACTOR
    zp1 = 1+z
    Ode = 1-Om
    mass_term = Om * zp1**3.
    de_term = Ode * zp1**(3.*(1.+w0+wa)) * np.exp(-3. * wa * z/zp1)
    Ez = (mass_term + de_term)**0.5

    value = 1./(H0_reduced * zp1 * Ez)

    return value

@nb.cfunc(quadpack_sig)
def lookback_time_integral(z, args):
    _args = nb.carray(args, (4,))
    value = lookback_time_integrand(z, _args)

    return value
lookback_time_integral_ptr = lookback_time_integral.address

@nb.jit
def dtd_prompt(
    t,
    H0,
    eta,
):
    h = H0/70.
    return eta * h**2

@nb.jit
def dtd_delayed(
    t,
    H0,
    eta,
):
    h = H0/70.
    return eta * h**2 * t**-1 

@nb.jit
def SFH(z, H0):
    
    h = H0/70.
    
    numerator = 0.015 * (1+z)**2.7
    denomenator = 1+((1+z) / 2.9)**5.6

    return h * numerator / denomenator 

@nb.jit
def N_prompt_integrand(zp, args):
    z, eta, H0 = args[:3]
    lookback_time_args = args[2:]

    jacobian = lookback_time_integrand(zp, lookback_time_args)
    value = SFH(zp, H0) * dtd_prompt(zp, H0, eta) * jacobian

    return value

@nb.jit
def N_delayed_integrand(zp, args):
    z, eta, H0 = args[:3]
    lookback_time_args = args[2:]

    tau, _, _ = dqags(lookback_time_integral_ptr, z, zp, lookback_time_args)
    jacobian = lookback_time_integrand(zp, lookback_time_args)
    value = SFH(zp, H0) * dtd_delayed(tau, H0, eta) * jacobian

    return value

@nb.cfunc(quadpack_sig)
def N_prompt_integral(zp, args):
    _args = nb.carray(args, (6,))
    value = N_prompt_integrand(zp, _args)

    return value
N_prompt_integral_ptr = N_prompt_integral.address

@nb.cfunc(quadpack_sig)
def N_delayed_integral(zp, args):
    _args = nb.carray(args, (6,))
    value = N_delayed_integrand(zp, _args)

    return value
N_delayed_integral_ptr = N_delayed_integral.address

@nb.jit
def volumetric_rates(
    z, integral_limits,
    H0, Om0, w0, wa,
    eta_prompt, eta_delayed,
    zinf=20.
):
    
    
    rates = np.zeros((len(z), 3), dtype=np.float64)
    for i in range(len(z)):

        zi = z[i]
        z0 = integral_limits[i, 0]
        z1 = integral_limits[i, 1]
        prompt_args = np.array([zi, eta_prompt, H0, Om0, w0, wa], dtype=np.float64)
        delayed_args = np.array([zi, eta_delayed, H0, Om0, w0, wa], dtype=np.float64)

        N_prompt, _, _ = dqags(
            N_prompt_integral_ptr, z0, z1, prompt_args
        )
        N_delayed, _, _ = dqags(
            N_delayed_integral_ptr, z1, zinf, delayed_args
        )

        rates[i, 0] = N_prompt + N_delayed
        rates[i, 1] = N_prompt
        rates[i, 2] = N_delayed
    
    return rates

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
                value += utils.uniform(
                    par_dict[value_key], **prep.global_model_cfg.prior_bounds[bounds_key]
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
        cov[:,:,0,0] += np.tile(
            z**(-2) * (
                (5. / np.log(10.))**2
                * (disp_v_pec / c)**2
            ), (2, 1)
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
        #cosmo: apy_cosmo.Cosmology
        H0: float, Om0: float, w0: float, wa: float
    ) -> np.ndarray:
        
        #global sn_observables
        mb = prep.sn_observables[:, 0]
        s = prep.sn_observables[:, 1]
        c = prep.sn_observables[:, 2]
        z = prep.sn_observables[:, 3]
        cosmo_params = (H0, Om0, 1.-Om0, w0, wa)

        residuals = np.zeros((2, len(mb), 3))

        distmod_values = np.array(
            distance_modulus_at_redshift(z, cosmo_params).tolist()
        )
        distance_moduli = np.tile(
            distmod_values, (2, 1)
        )
        #     cosmo.distmod(z).value, (2, 1)
        # ) #+ np.log10(cosmo.H0.value / H0)

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
            shift_Rb=shift_Rb
        )

        p_1 = probs[:, 0] / norm_1
        p_2 = probs[:, 1] / norm_2

        p1_nans = np.isnan(p_1)
        p2_nans = np.isnan(p_2)

        if np.any(p1_nans[:prep.idx_sn_to_evaluate]):
            print("Pop 1 contains nan probabilities:", np.count_nonzero(p1_nans)/len(p1_nans)*100, "%")
            print("Pop 1 pars:", [Rb_1, sig_Rb_1, Ebv_1, gamma_Ebv_1])
            print("Pop 1 norm:", norm_1, "\n")
        if np.any(p2_nans[:prep.idx_sn_to_evaluate]):
            print("Pop 2 contains nan probabilities:", np.count_nonzero(p2_nans)/len(p2_nans)*100, "%")
            print("Pop 1 pars:", [Rb_2, sig_Rb_2, Ebv_2, gamma_Ebv_2])
            print("Pop 2 norm:", norm_2, "\n")

        return p_1, p_2, status

    def independent_mvgaussian(
        self, means: np.ndarray, sigmas: np.ndarray
    ):

        n_properties = prep.host_galaxy_observables.shape[1]
        res = np.atleast_3d(
            np.where(
                prep.host_galaxy_observables == NULL_VALUE,
                0.,
                prep.host_galaxy_observables - means
            )
        )

        sigmas = np.tile(
            sigmas, [prep.host_galaxy_observables.shape[0], 1]
        )
        sigmas = np.where(
            prep.host_galaxy_observables == NULL_VALUE,
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
        w0: float, wa: float, eta_prompt: float,
        eta_delayed: float
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
            H0=H0, Om0=Om0, w0=w0, wa=wa
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

        if np.any(sn_probs_1 < 0.) | np.any(sn_probs_2 < 0.):
            print("\nOh no, someones below 0\n")
            return (
                -np.inf,
                np.ones(len(prep.sn_observables))*np.nan,
                np.ones(len(prep.sn_observables))*np.nan,
                np.ones(len(prep.sn_observables))*np.nan,
            )
        
        use_physical_population_fraction = (
            eta_delayed != NULL_VALUE and
            eta_prompt != NULL_VALUE and
            w == NULL_VALUE
        )

        if use_physical_population_fraction:
            dtd_t0 = prep.global_model_cfg['dtd_cfg']['t0']
            dtd_t1 = prep.global_model_cfg['dtd_cfg']['t1']
            z = prep.sn_observables[:, 3]
            convolution_time_limits = convolution_limits(
                cosmo, z, dtd_t0, dtd_t1
            )
            minimum_convolution_time = np.min(convolution_time_limits)
            z0 = apy_cosmo.z_at_value(cosmo.lookback_time, minimum_convolution_time * Gyr)
            H0_gyrm1 = cosmo.H0.to(1/Gyr).value
            cosmo_args = (H0_gyrm1, Om0, w0, wa)
            ts, zs, _ = redshift_at_times(
                convolution_time_limits, minimum_convolution_time, z0, cosmo_args
            )
            integral_limits = np.array(zs.tolist(), dtype=np.float64)
            sn_rates = volumetric_rates(
                z, integral_limits, cosmo.H0.value, Om0, w0, wa,
                eta_prompt, eta_delayed, zinf=20.
            )
            w_vector = sn_rates[:, -1] / sn_rates[:, 0]
        else:
            w_vector = np.ones_like(sn_probs_1) * w

        if host_galaxy_means.shape[0] > 0:
            host_probs_1, host_probs_2 = self.host_galaxy_probs(
                host_galaxy_means=host_galaxy_means,
                host_galaxy_sigmas=host_galaxy_sigs,
            )
            pop_1_probs = w_vector * sn_probs_1 * host_probs_1
            pop_2_probs = (1-w_vector) * sn_probs_2 * host_probs_2
            combined_probs = pop_1_probs + pop_2_probs
        else:
            host_probs_1 = np.ones(len(prep.sn_observables))*np.nan
            host_probs_2 = np.ones(len(prep.sn_observables))*np.nan
            pop_1_probs = w_vector * sn_probs_1
            pop_2_probs = (1-w_vector) * sn_probs_2
            combined_probs = pop_1_probs + pop_2_probs
        
        log_host_membership_probs = (
            1./np.log(10) * (
                np.log(w_vector * host_probs_1) - np.log((1-w_vector) * host_probs_2)
            ).flatten()
        )
        log_sn_membership_probs = (
            1./np.log(10) * (
                np.log(w_vector * sn_probs_1) - np.log((1-w_vector) * sn_probs_2)
            ).flatten()
        )
        log_full_membership_probs = 1./np.log(10) * (np.log(pop_1_probs) - np.log(pop_2_probs)).flatten()
        log_prob = np.sum(np.log(combined_probs[:prep.idx_sn_to_evaluate]))
        if np.isnan(log_prob):
            log_prob = -np.inf
            log_full_membership_probs = np.ones(len(prep.sn_observables))*np.nan
            log_host_membership_probs = np.ones(len(prep.sn_observables))*np.nan
            log_sn_membership_probs = np.ones(len(prep.sn_observables))*np.nan

        s1, s2 = np.all(status[:prep.idx_sn_to_evaluate, 0]), np.all(status[:prep.idx_sn_to_evaluate, 1])
        if not s1 or not s2:
            mean1, mean2 = np.mean(sn_probs_1), np.mean(sn_probs_2)
            std1, std2 = np.std(sn_probs_1), np.std(sn_probs_2)
            f1, f2 = np.count_nonzero(~status[:, 0])/len(status), np.count_nonzero(~status[:, 1])/len(status)
            print("\nPop 1 mean and std, percentage failed:", mean1, "+-", std1, ",", f1*100, "%")
            print("Pop 2 mean and std, percentage failed:", mean2, "+-", std2, ",", f2*100, "%")
            print("Log prob:", log_prob, "\n")
        
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
        )

        log_prior = self.log_prior(param_dict)
        if np.isinf(log_prior):
            return (
                log_prior,
                np.ones(len(prep.sn_observables))*np.nan,
                np.ones(len(prep.sn_observables))*np.nan,
                np.ones(len(prep.sn_observables))*np.nan,
            )
        
        if prep.global_model_cfg.use_sigmoid:
            param_dict = utils.apply_sigmoid(
                param_dict, sigmoid_cfg=prep.global_model_cfg.sigmoid_cfg,
                independent_par_names=prep.global_model_cfg.independent_par_names,
                ratio_par_name=prep.global_model_cfg.ratio_par_name
            )
        
        param_dict =  {
            **param_dict, **prep.global_model_cfg.preset_values,
        }
        (
            log_likelihood, log_full_membership_probs,
            log_sn_membership_probs, log_host_membership_probs
        ) = self.log_likelihood(**param_dict)

        return (
            log_likelihood, log_full_membership_probs,
            log_sn_membership_probs, log_host_membership_probs
        )
    