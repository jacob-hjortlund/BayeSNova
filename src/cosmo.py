import warnings
import numba as nb
import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.cosmology as cosmo

from base import Model
from astropy.units import Gyr
from numbalsoda import lsoda_sig, lsoda, dop853

@nb.njit
def nu_relative_density(
    z: np.ndarray, massive_nu: float,
    N_eff: float, nu_y: float, n_massless_nu: float,
    N_eff_per_nu: float
) -> np.ndarray:

    prefac = 0.22710731766 
    p = 1.83
    invp = 0.54644808743
    k = 0.3173

    if not massive_nu:

        relative_density = prefac * N_eff * (
            np.ones(z.shape) if hasattr(z, "shape") else 1.0
        )
    
    else:
        
        curr_nu_y = nu_y / (1 + np.array([z]))
        relative_mass_per = (
            (1.0 + (k * curr_nu_y)**p)**invp
        )
        relative_mass = relative_mass_per.sum(-1) + n_massless_nu

        relative_density = prefac * N_eff_per_nu * relative_mass

    return relative_density

@nb.njit
def de_density_scale(
    z: np.ndarray, w0: float, wa: float
) -> np.ndarray:

    zp1 = 1 + z
    de_term = zp1**(3.*(1.+w0+wa)) * np.exp(-3. * wa * z/zp1)

    return de_term

@nb.njit
def E(
    z: np.ndarray, args: np.ndarray
) -> np.ndarray:
    """The E(z) function for a Lambda + mattter universe.

    Args:
        z (np.ndarray): The redshift.
        args (np.ndarray): Array containing H0, Om0, w0, wa.

    Returns:
        np.ndarray: The E(z) function evaluated at z.
    """
    
    (
        t_H, Ogamma0, Onu0,
        Om0, Ode0, massive_nu,
        N_eff, nu_y, n_massless_nu,
        N_eff_per_nu,
        w0, wa
    ) = args

    nu_y = np.array(nu_y)

    zp1 = 1 + z

    Or = Ogamma0 + (
            Onu0
            if not massive_nu
            else Ogamma0 * nu_relative_density(
                z, massive_nu, N_eff,
                nu_y, n_massless_nu,
                N_eff_per_nu
            )
        )
    
    radiation_term = Or * zp1**4.
    mass_term = Om0 * zp1**3.
    de_term = Ode0 * de_density_scale(z, w0, wa)
    Ez = np.sqrt( radiation_term + mass_term + de_term)

    return Ez

#@nb.cfunc(lsoda_sig)
def z_ode(
    t, z, dz, cosmology_args
):
    """The RHS of ODE for the redshift at times.

    Args:
        t (float): The time, unused since RHS doesnt depend on t.
        z (np.ndarray): Array of redshifts.
        cosmology_args (np.ndarray): Array containing H0, Om0, w0, wa.

    Returns:
        np.ndarray: The ODE at the given redshift.
    """

    cosmology_args = nb.carray(cosmology_args, (12,))
    z = nb.carray(z, (1,))

    t_H0 = cosmology_args[0]
    zp1 = 1+z
    Ez = E(z, cosmology_args)
    dz[0] = -1 * t_H0 / ( zp1 * Ez)
z_ode_ptr = z_ode.address

def redshift_at_times(
    evaluation_times: np.ndarray,
    z0: float, t_H: float, Ogamma0: float,
    Onu0: float, Om0: float, Ode0: float,
    massive_nu: float, N_eff: float, nu_y: float,
    n_massless_nu: float, N_eff_per_nu: float,
    w0: float, wa: float
):
    
    z0 = np.array([z0])
    cosmology_args = np.array(
        [
            t_H, Ogamma0, Onu0,
            Om0, Ode0, massive_nu,
            N_eff, nu_y, n_massless_nu,
            N_eff_per_nu,
            w0, wa
        ]
    )

    usol, success = lsoda(
        z_ode_ptr, z0, t_eval=evaluation_times, data=cosmology_args
    )

    return usol, success

class Cosmology(Model):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        Ode0: float = 0.7,
        Tcmb0: float = 2.725, #update to planck value
        Neff: float = 3.046, #update to planck value
        m_nu: np.ndarray = np.array([0, 0, 0.06]), #update to planck value
        Ob0: float = 0.04897,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = Ode0
        self.Tcmb0 = Tcmb0
        self.Neff = Neff
        self.m_nu = m_nu
        self.Ob0 = Ob0
        self.w0 = -1.0
        self.wa = 0.0

        self.cosmo = cosmo.LambdaCDM(
            H0=self.H0, Om0=self.Om0, Ode0=self.Ode0,
            Tcmb0=self.Tcmb0, Neff=self.Neff, m_nu=self.m_nu,
            Ob0=self.Ob0,
        )

        self.t_H = self.cosmo.hubble_time.to_value("Gyr")

    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def E(self, z: np.ndarray) -> np.ndarray:
        
        args = np.array(
            [
                self.cosmo._Ogamma0, self.cosmo._Onu0,
                self.Om0, self.Ode0,
                self.cosmo._massivenu, self.cosmo._Neff,
                self.cosmo._nu_y[0], self.cosmo._nmasslessnu,
                self.cosmo._neff_per_nu, self.w0, self.wa
            ]
        )
        return E(z, args)

    def redshift_at_times(self, evaluation_times: np.ndarray) -> np.ndarray:
        
        z0 = "placeholder"
        usol, success = redshift_at_times(
            evaluation_times, z0, self.cosmo._Ogamma0, 
            self.cosmo._Onu0, self.Om0, self.Ode0,
            self.cosmo._massivenu, self.cosmo._Neff,
            self.cosmo._nu_y[0], self.cosmo._nmasslessnu,
            self.cosmo._neff_per_nu, self.w0, self.wa
        )

        return usol, success

class FlatLambdaCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        Tcmb0: float = 2.725, #update to planck value
        Neff: float = 3.046, #update to planck value
        m_nu: np.ndarray = np.array([0, 0, 6]), #update to planck value
        Ob0: float = 0.04897,
    ):
        
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=1-Om0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
        )
        self.cosmo = cosmo.FlatLambdaCDM(
            H0=self.H0, Om0=self.Om0,
            Tcmb0=self.Tcmb0, Neff=self.Neff,
            m_nu=self.m_nu, Ob0=self.Ob0,
        )
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value

class FlatwCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        Tcmb0: float = 2.725, #update to planck value
        Neff: float = 3.046, #update to planck value
        m_nu: np.ndarray = np.array([0, 0, 6]), #update to planck value
        Ob0: float = 0.04897,
        w0: float = -1.0,
    ):
        
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=1-Om0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
        )
        self.w0 = w0
        self.cosmo = cosmo.FlatwCDM(
            H0=self.H0, Om0=self.Om0,
            Tcmb0=self.Tcmb0, Neff=self.Neff,
            m_nu=self.m_nu, Ob0=self.Ob0,
            w0=self.w0,
        )
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value

class Flatw0waCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        Tcmb0: float = 2.725, #update to planck value
        Neff: float = 3.046, #update to planck value
        m_nu: np.ndarray = np.array([0, 0, 6]), #update to planck value
        Ob0: float = 0.04897,
        w0: float = -1.0,
        wa: float = 0.0,
    ):
        
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=1-Om0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
        )
        self.w0 = w0
        self.wa = wa
        self.cosmo = cosmo.Flatw0waCDM(
            H0=self.H0, Om0=self.Om0,
            Tcmb0=self.Tcmb0, Neff=self.Neff,
            m_nu=self.m_nu, Ob0=self.Ob0,
            w0=self.w0, wa=self.wa,
        )
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value