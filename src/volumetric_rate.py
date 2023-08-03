import warnings
import numpy as np
import astropy.cosmology as cosmo

from base import Model
from astropy.units import Gyr
from cosmo import Cosmology

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
        self.cosmology = cosmology
    
    def convolution_limits(
        self, z: np.ndarray, T0: float, T1: float,
        **kwargs
    ) -> np.ndarray:
        """The convolution limits for the redshift at times ODE.

        Args:
            z (np.ndarray): Redshifts.
            T0 (float): The lower time limit in Gyrs.
            T1 (float): The upper time limit in Gyrs.
            cosmology (cosmo.Cosmology, optional): The AstroPy cosmology. Defaults to None.
            H0 (float, optional): The Hubble constant. Defaults to None.
            Om0 (float, optional): The matter density. Defaults to None.
            w0 (float, optional): The dark energy equation of state. Defaults to None.
            wa (float, optional): The dark energy equation of state evolution. Defaults to None.

        Returns:
            jax.Array: The convolution limits.
        """

        self.check_if_initialized()
        
        times_z0_lower = self.cosmology.cosmology.age(z).value - T0
        times_z0_upper = self.cosmology.cosmology.age(z).value - T1

        return np.concatenate((times_z0_lower, times_z0_upper))