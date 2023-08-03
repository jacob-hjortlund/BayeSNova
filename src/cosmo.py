import warnings
import numba as nb
import numpy as np
import astropy.cosmology as cosmo

from base import Model
from astropy.units import Gyr
from numbalsoda import lsoda_sig, lsoda, dop853

@nb.njit
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

    _, _, Om0, w0, wa = args
    Ode0 = 1-Om0
    zp1 = 1+z
    mass_term = Om0 * zp1**3.
    de_term = Ode0 * zp1**(3.*(1.+w0+wa)) * np.exp(-3. * wa * z/zp1)
    Ez = np.sqrt(mass_term + de_term)

    return Ez

@nb.cfunc(lsoda_sig)
def z_ode(
    t, z, dz, cosmology_args
):
    """The RHS of ODE for the redshift at times.

    Args:
        t (float): The time, unused since RHS doesnt depend on t.
        z (np.ndarray): Array of redshifts.
        cosmology_args (np.ndarray): Array containing H0, Om0, w0, wa.

    Returns:
        ArrayLike: The ODE at the given redshift.
    """

    cosmology_args = nb.carray(cosmology_args, (5,))
    z = nb.carray(z, (1,))

    t_H0, _, _, _, _ = cosmology_args
    zp1 = 1+z
    Ez = E(z, cosmology_args)
    dz[0] = -1. / t_H0 * zp1 * Ez
z_ode_ptr = z_ode.address

def redshift_at_times(
    evaluation_times: np.ndarray,
    z0: float, cosmology_args: np.ndarray
):
    
    z0 = np.array([z0])
    usol, success = lsoda(
        z_ode_ptr, z0, t_eval=evaluation_times, data=cosmology_args
    )

    return usol, success

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