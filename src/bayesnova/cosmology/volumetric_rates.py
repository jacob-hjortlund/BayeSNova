import numpy as np
import numba as nb

from astropy.cosmology import Flatw0waCDM
from NumbaQuadpack import quadpack_sig, dqags, ldqag

import src.bayesnova.utils.constants as constants
import src.bayesnova.cosmology.cosmology as cosmology

# ---------- N(z) integral ------------

@nb.jit()
def prompt_frac_to_prompt_rate(
    prompt_frac: float, delayed_rate: float,
    t_max: float, t0: float = 0.04, t1: float = 0.5
) -> float:
    """Converts the prompt progenitor fraction to the prompt progenitor rate.

    Args:
        prompt_frac (float): The prompt progenitor fraction.
        delayed_rate (float): The delayed progenitor rate.
        t_max (float): The maximum delay time for SNe in Gyrs.
        t0 (float, optional): The minimum delay time for prompt SNe in Gyrs. Defaults to 0.04.
        t1 (float, optional): The maximum delay time for prompt SNe in Gyrs. Defaults to 0.5.

    Returns:
        float: The prompt progenitor rate.
    """
    
    prompt_rate = (
        delayed_rate *
        prompt_frac / (1. - prompt_frac) *
        (np.log(t_max) - np.log(t1)) / (t1 - t0)
    )

    return prompt_rate

@nb.jit()
def lookback_time_integrand(
    z: float, args: np.ndarray
) -> float:
    """The integrand for the lookback time integral.

    Args:
        z (float): The redshift.
        args (np.ndarray): Array containing H0, Om, w0, wa.

    Returns:
        float: The integrand at the given redshift.
    """

    H0, Om, w0, wa = args
    H0_reduced = H0 * constants.H0_CONVERSION_FACTOR
    zp1 = 1+z
    Ode = 1-Om
    mass_term = Om * zp1**3.
    de_term = Ode * zp1**(3.*(1.+w0+wa)) * np.exp(-3. * wa * z/zp1)
    Ez = (mass_term + de_term)**0.5

    value = 1./(H0_reduced * zp1 * Ez)

    return value

@nb.cfunc(quadpack_sig)
def lookback_time_integral(z, args):
    """The lookback time integral.

    Args:
        z (float): The redshift.
        args: Array containing H0, Om, w0, wa.

    Returns:
        float: The lookback time at the given redshift.
    """
    
    _args = nb.carray(args, (4,))
    value = lookback_time_integrand(z, _args)

    return value
lookback_time_integral_ptr = lookback_time_integral.address

@nb.jit()
def dtd_prompt(
    t: float,
    H0: float,
    eta: float,
) -> float:
    """The prompt progenitor channel delay-time distribution.

    Args:
        t (float): The delay time in Gyrs.
        H0 (float): The Hubble constant in km/s/Mpc.
        eta (float): The SN Ia normalization.

    Returns:
        float: The delay-time distribution at the given delay time.
    """

    h = H0/70.
    return eta * h**2

@nb.jit()
def dtd_delayed(
    t: float,
    H0: float,
    eta: float,
) -> float:
    """The delayed progenitor channel delay-time distribution.

    Args:
        t (float): The delay time in Gyrs.
        H0 (float): The Hubble constant in km/s/Mpc.
        eta (float): The SN Ia normalization.

    Returns:
        float: The delay-time distribution at the given delay time.
    """

    h = H0/70.
    return eta * h**2 * t**-1 

@nb.jit()
def SFH(z: float, H0: float) -> float:
    """The cosmic star formation history from Madau et al 2014.

    Args:
        z (float): The redshift.
        H0 (float): The Hubble constant in km/s/Mpc.

    Returns:
        float: The star formation rate at the given redshift.
    """
    
    h = H0/70.
    
    numerator = 0.015 * (1+z)**2.7
    denomenator = 1+((1+z) / 2.9)**5.6

    return h * numerator / denomenator 

@nb.jit()
def N_prompt_integrand(zp: float, args: np.ndarray) -> float:
    """The integrand for the prompt progenitor channel volumetric rate integral.

    Args:
        zp (float): The delay redshift.
        args (np.ndarray): Array containing z, eta, H0, Om, w0, wa.

    Returns:
        float: The integrand at the given redshift.
    """

    z, eta, H0 = args[:3]
    lookback_time_args = args[2:]

    jacobian = lookback_time_integrand(zp, lookback_time_args)
    value = SFH(zp, H0) * dtd_prompt(zp, H0, eta) * jacobian

    return value

@nb.jit()
def N_delayed_integrand(zp: float, args: np.ndarray) -> float:
    """The integrand for the delayed progenitor channel volumetric rate integral.

    Args:
        zp (float): The delay redshift.
        args (np.ndarray): Array containing z, eta, H0, Om, w0, wa.

    Returns:
        float: The integrand at the given redshift.
    """

    z, eta, H0 = args[:3]
    lookback_time_args = args[2:]

    tau, _, _, _ = dqags(lookback_time_integral_ptr, z, zp, lookback_time_args)
    jacobian = lookback_time_integrand(zp, lookback_time_args)
    value = SFH(zp, H0) * dtd_delayed(tau, H0, eta) * jacobian

    return value

@nb.cfunc(quadpack_sig)
def N_prompt_integral(zp, args):
    """The prompt progenitor channel volumetric rate integral.

    Args:
        zp (float): The delay redshift.
        args: Array containing z, eta, H0, Om, w0, wa.

    Returns:
        float: The volumetric rate at the given redshift.
    """

    _args = nb.carray(args, (6,))
    value = N_prompt_integrand(zp, _args)

    return value
N_prompt_integral_ptr = N_prompt_integral.address

@nb.cfunc(quadpack_sig)
def N_delayed_integral(zp, args):
    """The delayed progenitor channel volumetric rate integral.

    Args:
        zp (float): The delay redshift.
        args: Array containing z, eta, H0, Om, w0, wa.

    Returns:
        float: The volumetric rate at the given redshift.
    """

    _args = nb.carray(args, (6,))
    value = N_delayed_integrand(zp, _args)

    return value
N_delayed_integral_ptr = N_delayed_integral.address

@nb.jit()
def _volumetric_rates(
    z: np.ndarray, integral_limits: np.ndarray,
    H0: float, Om0: float, w0: float, wa: float,
    eta: float, prompt_fraction: float,
    zinf: float, age: float
) -> np.ndarray:
    """Calculate the total, prompt and delayed progenitor channel volumetric rates at the given redshifts.

    Args:
        z (np.ndarray): The redshifts.
        integral_limits (np.ndarray): The limits of integration for each redshift.
        H0 (float): The Hubble constant in km/s/Mpc.
        Om0 (float): The matter density parameter.
        w0 (float): The dark energy equation of state parameter.
        wa (float): The dark energy equation of state parameter.
        eta (float): The SN Ia normalization.
        prompt_fraction (float): The prompt fraction.
        zinf (float): The maximum redshift of star formation.
        age (float): The age of the universe.

    Returns:
        np.ndarray: The total, prompt and delayed progenitor channel volumetric rates at the given redshifts.
    """
    
    eta_prompt = prompt_frac_to_prompt_rate(
        prompt_fraction, eta, age
    )
    rates = np.zeros((len(z), 3), dtype=np.float64)
    for i in range(len(z)):

        zi = z[i]
        z0 = integral_limits[i, 0]
        z1 = integral_limits[i, 1]
        prompt_args = np.array([zi, eta_prompt, H0, Om0, w0, wa], dtype=np.float64)
        delayed_args = np.array([zi, eta, H0, Om0, w0, wa], dtype=np.float64)

        N_prompt = N_delayed = 0.
        
        z0_valid = z0 != constants.NULL_VALUE
        z1_valid = z1 != constants.NULL_VALUE

        if z1_valid:
            N_delayed, _, _, _ = dqags(
                N_delayed_integral_ptr, z1, zinf, delayed_args
            )
        else:
            z1 = zinf

        if z0_valid:
            N_prompt, _, _, _ = dqags(
                N_prompt_integral_ptr, z0, z1, prompt_args
            )

        rates[i, 0] = N_prompt + N_delayed
        rates[i, 1] = N_prompt
        rates[i, 2] = N_delayed
    
    return rates

def volumetric_rates(
    z: np.ndarray, H0: float, Om0: float,
    w0: float, wa: float,
    eta: float, prompt_fraction: float,
    t0: float = 0.04, t1: float = 0.5,
    zinf: float = 20., **kwargs
) -> np.ndarray:
    """Calculate the total, prompt and delayed progenitor channel volumetric rates at the given redshifts.

    Args:
        z (np.ndarray): The redshifts.
        H0 (float): The Hubble constant in km/s/Mpc.
        Om0 (float): The matter density parameter.
        w0 (float): The dark energy equation of state parameter.
        wa (float): The dark energy equation of state parameter.
        eta (float): The SN Ia normalization.
        prompt_fraction (float): The prompt fraction.
        t0 (float, optional): The minimum delay time in Gyr. Defaults to 0.04.
        t1 (float, optional): The maximum delay time in Gyr. Defaults to 0.5.
        zinf (float, optional): The maximum redshift of star formation. Defaults to 20.

    Returns:
        np.ndarray: The total, prompt and delayed progenitor channel volumetric rates at the given redshifts.
    """
    
    cosmo = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
    cosmo_args = (H0, Om0, w0, wa)
    convolution_time_limits = cosmology.convolution_time_limits(
        z=z, T0=t0, T1=t1, cosmology=cosmo
    )
    idx_valid_limits = convolution_time_limits > 0.
    valid_convolution_time_limits = convolution_time_limits[idx_valid_limits]
    minimum_convolution_time = np.min(valid_convolution_time_limits)

    z0 = cosmology.initial_redshift_value(
        minimum_convolution_time, cosmology=cosmo
    )
    is_z0_valid = np.isfinite(z0) and z0 > 0.
    if not is_z0_valid:
        sn_ia_rates = np.ones((len(z), 3)) * -np.inf

    _, convolution_redshift_limits, _ = cosmology.redshift_at_times(
        times=convolution_time_limits, t0=minimum_convolution_time,
        z0=z0, cosmo_args=cosmo_args
    )

    integral_limits = np.ones_like(convolution_redshift_limits) * constants.NULL_VALUE
    integral_limits[idx_valid_limits] = convolution_redshift_limits[idx_valid_limits]
    integral_limits = np.column_stack(
        np.split(integral_limits, 2)
    )

    age_of_universe = cosmo.age(0).value - cosmo.age(zinf).value

    sn_ia_rates = _volumetric_rates(
        z=z, integral_limits=integral_limits, H0=H0, Om0=Om0,
        w0=w0, wa=wa, eta=eta, prompt_fraction=prompt_fraction,
        zinf=zinf, age=age_of_universe
    )

    return sn_ia_rates