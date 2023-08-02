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

    def initial_redshift_value(
        self, initial_time: float,
        z_at_value_kwargs: dict = {'method': 'Bounded', 'zmax': 1000},
        zmax: float = 1e10, factor: float = 10, n_repeats: int = 0, **kwargs
    ) -> float:
        """The initial redshift value for the redshift at times ODE.

        Args:
            initial_time (float): The initial time in Gyrs.
            cosmo_kwargs (dict, optional): The kwargs for the z_at_value function. Defaults to {'method': 'Bounded', 'zmax': 1000}.
            zmax (float, optional): The maximum redshift for z_at_value. Defaults to 1e10.
            factor (float, optional): The factor to increase z_at_value zmax by. Defaults to 10.
            n_repeats (int, optional): The number of times to repeat the z_at_value function. Defaults to 0.
            cosmology (cosmo.Cosmology, optional): The AstroPy cosmology. Defaults to None.
            H0 (float, optional): The Hubble constant. Defaults to None.
            Om0 (float, optional): The matter density. Defaults to None.
            w0 (float, optional): The dark energy equation of state. Defaults to None.
            wa (float, optional): The dark energy equation of state evolution. Defaults to None.

        Returns:
            float: The initial redshift value.
        """

        self.check_if_initialized()

        if z_at_value_kwargs['zmax'] > zmax:
            raise ValueError(
                'Upper limit for initial age to redshift ODE ' +
                'is above 1e10, something is wrong.'
            )

        warnings.filterwarnings('ignore')
        try:
            z0 = cosmo.z_at_value(
                self.cosmology.cosmology.age, initial_time * Gyr,
                **z_at_value_kwargs
            )
        except:
            warnings.resetwarnings()
            warning_str = (
                f"Failure to find z0 for minimum convolution time of {initial_time} Gyr."
            )
            if n_repeats > 0:
                warning_str += f" Trying again with zmax = {zmax * factor}."
                warnings.warn(warning_str)
                z_at_value_kwargs['zmax'] *= factor
                z0 = self.initial_redshift_value(
                    initial_time=initial_time,
                    z_at_value_kwargs=z_at_value_kwargs,
                    zmax=zmax, factor=factor
                )
            else:
                warnings.warn(warning_str)
                z0 = np.nan

        return z0