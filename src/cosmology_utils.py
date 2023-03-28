import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import numba as nb
import jax.numpy as jnp
import astropy.cosmology as apy_cosmo

from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

H0_CONVERSION_FACTOR = 0.001022
DH_70 = 4282.7494

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

def initial_value(cosmo, time, init_limit=1000):

    if init_limit > 1e10:
        raise ValueError(
            'Upper limit for initial age to redshift ODE ' +
            'is above 1e10, something is wrong.'
        )

    try:
    
        z0 = apy_cosmo.z_at_value(
            cosmo.age, time * Gyr, zmax=init_limit,
            method='Bounded'
        )
    except:
        z0 = initial_value(cosmo, time, init_limit=init_limit*10)

    return z0

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
        args=cosmo_args, max_steps=4096*16
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

def prompt_frac_to_prompt_rate(
    prompt_frac, delayed_rate,
    t_max, t0 = 0.04, t1 = 0.5
):
    
    prompt_rate = (
        delayed_rate *
        prompt_frac / (1. - prompt_frac) *
        (np.log(t_max) - np.log(t1)) / (t1 - t0)
    )

    return prompt_rate

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
