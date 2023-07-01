import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import astropy.cosmology as cosmo
import src.bayesnova.utils.constants as constants

from astropy.units import Gyr
from jax.typing import ArrayLike
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

# ---------- z(T) ODE ------------

def E(
    z: ArrayLike, args: ArrayLike
) -> ArrayLike:
    """The E(z) function for the cosmology.

    Args:
        z (ArrayLike): The redshift.
        args (ArrayLike): Array containing H0, Om0, w0, wa.

    Returns:
        ArrayLike: The E(z) function evaluated at z.
    """

    _, Om0, w0, wa = args
    Ode0 = 1-Om0
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
        args (ArrayLike): Array containing H0, Om0, w0, wa.

    Returns:
        ArrayLike: The ODE at the given redshift.
    """

    H0, _, _, _ = args
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
        cosmo_args (ArrayLike): Array containing H0, Om0, w0, wa.

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
        cosmo_args (ArrayLike): Array containing H0, Om0, w0, wa.

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
        cosmo_args (ArrayLike): Array containing H0, Om0, w0, wa.

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
