import warnings
import numpy as np
import astropy.cosmology as cosmo

from base import Model
from astropy.units import Gyr


def E(
    z: np.ndarray, args: np.ndarray
) -> np.ndarray:
    """The E(z) function for a Lambda + mattter universe.

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
    de_term = Ode0 * zp1**(3.*(1.+w0+wa)) * np.exp(-3. * wa * z/zp1)
    Ez = np.sqrt(mass_term + de_term)

    return Ez

def ode(
    t: float, z: np.ndarray, args: np.ndarray
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

class Cosmology(Model):

    def __init__(
        self,
    ):
        
        super().__init__()
        self.cosmology = None

    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def check_if_initialized(self) -> None:
        if self.cosmology is None:
            raise ValueError('Cosmology has not been initialized.')

class FlatLambdaCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.cosmo = cosmo.FlatLambdaCDM(H0=self.H0, Om0=self.Om0)
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value

class FlatwCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        w0: float = -1.0,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = w0
        self.cosmo = cosmo.FlatwCDM(H0=self.H0, Om0=self.Om0, w0=self.w0)
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value

class Flatw0waCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        w0: float = -1.0,
        wa: float = 0.0,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = w0
        self.wa = wa
        self.cosmo = cosmo.Flatw0waCDM(H0=self.H0, Om0=self.Om0, w0=self.w0, wa=self.wa)
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value