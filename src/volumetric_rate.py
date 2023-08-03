import warnings
import numba as nb
import numpy as np
import astropy.cosmology as cosmo

from base import Model
from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags
from cosmo import Cosmology, lookback_time, lookback_time_integrand

def _cosmic_star_formation_history(
    z: np.ndarray, H0: float
) -> np.ndarray:
    
    h = H0 / 70
    numerator = 0.015 * (1 + z)**2.7 * h
    denominator = 1 + ((1 + z) / 2.9)**5.6
    sfh = numerator / denominator

    return sfh

def _prompt_delay_time_distribution(
    tau: np.ndarray, eta: float, f_prompt: float, H0: float, 
    tau_0: float, tau_1: float, tau_max: float
) -> np.ndarray:

    h = H0 / 70
    K = np.log( tau_max / tau_1 ) / (tau_1 - tau_0) 
    f_frac = f_prompt / (1 - f_prompt)
    dtd = (K * eta * f_frac * h**2 )* np.ones_like(tau)

    return dtd

def _delayed_delay_time_distribution(
    tau: np.ndarray, eta: float, f_prompt: float, H0: float,
    tau_0: float, tau_1: float, tau_max: float
) -> np.ndarray:
    
    h = H0 / 70
    dtd = eta / tau * h**2

    return dtd

def _prompt_rate_integrand(zt, args):

    z, eta, f_prompt, tau_0, tau_1, tau_max = args[:6]
    cosmology_args = args[6:]
    H0 = cosmology_args[1]

    SFH = _cosmic_star_formation_history(z, H0)
    prompt_dtd = _prompt_delay_time_distribution(
        tau=zt, eta=eta, f_prompt=f_prompt, H0=H0,
        tau_0=tau_0, tau_1=tau_1, tau_max=tau_max
    )
    jacobian = lookback_time_integrand(
        zt, cosmology_args
    )

    value = SFH * prompt_dtd * jacobian

    return value

def _delayed_rate_integrand(zt, args):

    z, eta, f_prompt, tau_0, tau_1, tau_max = args[:6]
    cosmology_args = args[6:]
    (
        t_H, H0, Ogamma0, Onu0,
        Om0, Ode0, massive_nu,
        N_eff, nu_y, n_massless_nu,
        N_eff_per_nu,
        w0, wa
    ) = cosmology_args

    tau = lookback_time(
        z, zt, t_H, H0, Ogamma0, Onu0,
        Om0, Ode0, massive_nu,
        N_eff, nu_y, n_massless_nu,
        N_eff_per_nu, w0, wa
    )
    SFH = _cosmic_star_formation_history(z, H0)
    delayed_dtd = _delayed_delay_time_distribution(
        tau=tau, eta=eta, f_prompt=f_prompt, H0=H0,
        tau_0=tau_0, tau_1=tau_1, tau_max=tau_max
    )
    jacobian = lookback_time_integrand(
        zt, cosmology_args
    )

    value = SFH * delayed_dtd * jacobian

    return value

@nb.cfunc(quadpack_sig)
def _prompt_rate_integral(
    zt, args
):
    
    args = nb.carray(args, (19,))
    zt = nb.carray(zt, (1,))

    integral_value = _prompt_rate_integrand(zt, args)

    return integral_value
prompt_rate_integral_ptr = _prompt_rate_integral.address

@nb.cfunc(quadpack_sig)
def _delayed_rate_integral(
    zt, args
):
    
    args = nb.carray(args, (19,))
    zt = nb.carray(zt, (1,))

    integral_value = _delayed_rate_integrand(zt, args)

    return integral_value
delayed_rate_integral_ptr = _delayed_rate_integral.address

def _volumetric_rates(
    z: np.ndarray, integral_limits: np.ndarray, 
    eta: float, f_prompt: float,
    tau_0: float, tau_1: float, tau_max: float,
    z_inf: float, cosmology_args: np.ndarray
) -> np.ndarray:
    
    n_redshifts = len(z)
    rates = np.zeros((n_redshifts, 3))
    for i in range(n_redshifts):

        zi = z[i]
        z0 = integral_limits[i, 0]
        z1 = integral_limits[i, 1]
        args = np.concatenate(
            [
                [zi, eta, f_prompt, tau_0, tau_1, tau_max],
                cosmology_args
            ], dtype=np.float64
        )

        prompt_rate = delayed_rate = 0.

        z0_valid = z0 != np.nan
        z1_valid = z1 != np.nan

        if z0_valid:

            prompt_rate, _ = dqags(
                prompt_rate_integral_ptr, 
                z0, z1,
                args,
            )
        
        if z1_valid:

            delayed_rate, _ = dqags(
                delayed_rate_integral_ptr, 
                z1, z_inf,
                args,
            )
        
        rates[i, 0] = prompt_rate
        rates[i, 1] = delayed_rate
        rates[i, 2] = prompt_rate + delayed_rate

    return rates

class SNeProgenitors(Model):

    def __init__(
        self,
        cosmology: Cosmology,
        eta: float = 1.02e-4,
        f_prompt: float = 0.63,
    ):
        
        super().__init__()
        self.eta = eta
        self.f_prompt = f_prompt
        self.cosmo = cosmology
    
    def convolution_limits(
        self, z: np.ndarray, tau_0: float, tau_1: float,
    ) -> np.ndarray:
        """The convolution limits for the redshift at times ODE.

        Args:
            z (np.ndarray): Redshifts.
            T0 (float): The lower time limit in Gyrs.
            T1 (float): The upper time limit in Gyrs.

        Returns:
            np.ndarray: The convolution limits.
        """
        
        times_z0_lower = self.cosmo.cosmo.age(z).value - tau_0
        times_z0_upper = self.cosmo.cosmo.age(z).value - tau_1

        return np.concatenate((times_z0_lower, times_z0_upper))
    
    def volumetric_rates(
        self, z: np.ndarray, tau_0: float, tau_1: float,
        z_inf: float = 50.
    ) -> np.ndarray:
        
        convolution_time_limits = self.convolution_limits(
            z, tau_0, tau_1
        )
        idx_valid = convolution_time_limits > 0.
        convolution_time_limits = convolution_time_limits[idx_valid]

        redshifts_at_convolution_time_limits, ode_status = self.cosmo.redshift_at_times(
            convolution_time_limits
        )

        if not ode_status:
            raise ValueError(
                "ODE solver failed to converge for some redshifts."
            )
        
        convolution_redshift_limits = np.ones(2*len(z)) * np.nan
        convolution_redshift_limits[idx_valid] = redshifts_at_convolution_time_limits 
        convolution_redshift_limits = np.column_stack(
            np.split(
                convolution_redshift_limits, 2
            )
        )
        
        volumetric_rates = _volumetric_rates(
            z=z, integral_limits=convolution_redshift_limits,
            eta=self.eta, f_prompt=self.f_prompt,
            tau_0=tau_0, tau_1=tau_1, tau_max=self.cosmo.cosmo.age(0).value,
            z_inf=z_inf, cosmology_args=self.cosmo.cosmology_args
        )

        return volumetric_rates


        
        