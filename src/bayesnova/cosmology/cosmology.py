import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import numba as nb

import jax.numpy as jnp
import astropy.cosmology as cosmo

from astropy.units import Gyr
from jax.typing import ArrayLike
from NumbaQuadpack import quadpack_sig, dqags, ldqag
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

import src.bayesnova.utils.constants as constants

# ---------- z(T) ODE ------------

def E(
    z: ArrayLike, args: ArrayLike
) -> ArrayLike:
    """The E(z) function for the cosmology.

    Args:
        z (ArrayLike): The redshift.
        args (ArrayLike): Array containing H0, Om0, Ode0, w0, wa.

    Returns:
        ArrayLike: The E(z) function evaluated at z.
    """

    _, Om0, Ode0, w0, wa = args
    zp1 = 1+z
    mass_term = Om0 * zp1**3.
    de_term = Ode0 * zp1**(3.*(1.+w0+wa)) * jnp.exp(-3. * wa * z/zp1)
    Ez = jnp.sqrt(mass_term + de_term)

    return Ez

def convolution_limits(
    cosmology: cosmo.Cosmology, z: np.ndarray, T0: float, T1: float
) -> jax.Array:
    """The convolution limits for the redshift at times ODE.

    Args:
        cosmology (cosmo.Cosmology): The AstroPy cosmology.
        z (np.ndarray): Redshifts.
        T0 (float): The lower time limit in Gyrs.
        T1 (float): The upper time limit in Gyrs.

    Returns:
        jax.Array: The convolution limits.
    """
    
    times_z0_lower = cosmology.age(z).value - T0
    times_z0_upper = cosmology.age(z).value - T1

    return jnp.concatenate((times_z0_lower, times_z0_upper))

def initial_redshift_value(
    cosmology: cosmo.Cosmology, initial_time: float,
    cosmo_kwargs: dict = {'method': 'Bounded', 'zmax': 1000},
    zmax: float = 1e10, factor: float = 10
) -> float:
    """The initial redshift value for the redshift at times ODE.

    Args:
        cosmology (cosmo.Cosmology): The AstroPy cosmology.
        initial_time (float): The initial time in Gyrs.
        cosmo_kwargs (dict, optional): The kwargs for the z_at_value function. Defaults to {'method': 'Bounded', 'zmax': 1000}.
        zmax (float, optional): The maximum redshift for z_at_value. Defaults to 1e10.
        factor (float, optional): The factor to increase z_at_value zmax by. Defaults to 10.

    Returns:
        float: The initial redshift value.
    """

    if cosmo_kwargs['zmax'] > zmax:
        raise ValueError(
            'Upper limit for initial age to redshift ODE ' +
            'is above 1e10, something is wrong.'
        )

    try:
        z0 = cosmo.z_at_value(
            cosmology.age, initial_time * Gyr, **cosmo_kwargs
        )
    except:
        cosmo_kwargs['zmax'] *= factor
        z0 = initial_redshift_value(
            cosmology, initial_time, cosmo_kwargs, zmax, factor
        )

    return z0

def ode(
    t: ArrayLike, z: ArrayLike, args: ArrayLike
):
    """The RHS of ODE for the redshift at times.

    Args:
        t (ArrayLike): The time, unused since RHS doesnt depend on t.
        z (ArrayLike): Array of redshifts.
        args (ArrayLike): Array containing H0, Om0, Ode0, w0, wa.

    Returns:
        ArrayLike: The ODE at the given redshift.
    """

    H0, _, _, _, _ = args
    zp1 = 1+z
    Ez = E(z, args)

    return -H0 * zp1 * Ez

@jax.jit
def redshift_at_times(
    conv_limit_times: jax.Array,
    t0: ArrayLike, z0: ArrayLike,
    cosmo_args: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """The redshift at times ODE.

    Args:
        conv_limit_times (jax.Array): The convolution limits in Gyrs.
        t0 (ArrayLike): The initial time in Gyrs.
        z0 (ArrayLike): The initial redshift.
        cosmo_args (ArrayLike): Array containing H0, Om0, Ode0, w0, wa.

    Returns:
        tuple[ArrayLike, ArrayLike, ArrayLike]: The redshift at times,
        the corresponding times, and the number of steps.
    """

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
        args=cosmo_args, max_steps=4096*16
    )

    ts = sol.ts[idx_unsort]
    zs = sol.ys[idx_unsort]

    num_steps = sol.stats['num_steps']

    return ts, zs, num_steps

# ---------- mu(z) via ODE ------------

def dc_ode(
    z: ArrayLike, D: ArrayLike, cosmo_args: ArrayLike
) -> ArrayLike:
    """The RHS of ODE for the comoving distance.

    Args:
        z (ArrayLike): Redshifts.
        D (ArrayLike): Comoving distance.
        cosmo_args (ArrayLike): Array containing H0, Om0, Ode0, w0, wa.

    Returns:
        ArrayLike: The ODE at the given redshift.
    """

    Ez = E(z, cosmo_args)

    return  1. / Ez

@jax.jit
def distance_modulus_at_redshift(
    z: ArrayLike, cosmo_args: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """The distance modulus at redshift ODE.

    Args:
        z (ArrayLike): The redshifts.
        cosmo_args (ArrayLike): Array containing H0, Om0, Ode0, w0, wa.

    Returns:
        tuple[ArrayLike, ArrayLike, ArrayLike]: The distance modulus at redshift,
        corresponding redshifts, and number of steps.
    """

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
    DH = constants.DH_70 * 70./H0
    zs = sol.ts[idx_unsort]
    dcs = sol.ys[idx_unsort] * DH
    dls = (1+z) * dcs
    mus = 5.0 * jnp.log10(dls) + 25.0
    
    num_steps = sol.stats['num_steps']

    return mus, zs, num_steps

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
def volumetric_rates(
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
