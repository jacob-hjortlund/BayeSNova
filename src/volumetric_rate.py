import warnings
import numba as nb
import numpy as np
import astropy.cosmology as cosmo

from base import Model
from cosmo import Cosmology
from astropy.units import Gyr

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
    K = np.log( ( tau_max / tau_1 ) / (tau_1 - tau_0) )
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
    lookback_time_args = args[6:8]
    cosmology_args = args[8:]
    H0 = cosmology_args[1]
    
    SFH = _cosmic_star_formation_history(z, H0)
    prompt_dtd = _prompt_delay_time_distribution(
        tau=zt, eta=eta, f_prompt=f_prompt, H0=H0,
        tau_0=tau_0, tau_1=tau_1, tau_max=tau_max
    )
    jacobian = 1.

    value = SFH * prompt_dtd * jacobian

    return value

def _delayed_rate_integrand(zt, args):

    z, eta, f_prompt, tau_0, tau_1, tau_max = args[:6]
    lookback_time_args = args[6:8]
    cosmology_args = args[8:]
    H0 = cosmology_args[1]
    
    SFH = _cosmic_star_formation_history(z, H0)
    delayed_dtd = _delayed_delay_time_distribution(
        tau=zt, eta=eta, f_prompt=f_prompt, H0=H0,
        tau_0=tau_0, tau_1=tau_1, tau_max=tau_max
    )
    jacobian = 1.

    value = SFH * delayed_dtd * jacobian

    return value


class VolumetricRate(Model):

    def __init__(
        self,
        eta: float,
        f_prompt: float,
        cosmology: Cosmology
    ):
        
        super().__init__()
        self.eta = eta
        self.f_prompt = f_prompt
        self.cosmo = cosmology
    
    def convolution_limits(
        self, z: np.ndarray, T0: float, T1: float,
    ) -> np.ndarray:
        """The convolution limits for the redshift at times ODE.

        Args:
            z (np.ndarray): Redshifts.
            T0 (float): The lower time limit in Gyrs.
            T1 (float): The upper time limit in Gyrs.

        Returns:
            np.ndarray: The convolution limits.
        """
        
        times_z0_lower = self.cosmo.cosmo.age(z).value - T0
        times_z0_upper = self.cosmo.cosmo.age(z).value - T1

        return np.concatenate((times_z0_lower, times_z0_upper))
    
    def cosmic_star_formation_history(self, z: np.ndarray) -> np.ndarray:
        pass