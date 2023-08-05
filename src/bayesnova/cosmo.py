import warnings
import numba as nb
import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.cosmology as cosmo

from bayesnova.base import Model
from astropy.units import Gyr
from NumbaQuadpack import quadpack_sig, dqags
from numbalsoda import lsoda_sig, lsoda, dop853

@nb.njit
def nu_relative_density(
    z: float, massive_nu: float,
    N_eff: float, nu_y: float, n_massless_nu: float,
    N_eff_per_nu: float
) -> float:

    prefac = 0.22710731766 
    p = 1.83
    invp = 0.54644808743
    k = 0.3173

    curr_nu_y = nu_y / (1 + z)
    relative_mass_per = (
        (1.0 + (k * curr_nu_y)**p)**invp
    )
    relative_mass = relative_mass_per + n_massless_nu

    relative_density = prefac * N_eff_per_nu * relative_mass

    return relative_density

@nb.njit
def de_density_scale(
    z: float, w0: float, wa: float
) -> float:

    zp1 = 1 + z
    de_term = zp1**(3.*(1.+w0+wa)) * np.exp(-3. * wa * z/zp1)

    return de_term

@nb.njit
def E(
    z: float, args: np.ndarray
) -> float:
    """The E(z) function for a Lambda + mattter universe.

    Args:
        z (np.ndarray): The redshift.
        args (np.ndarray): Array containing H0, Om0, w0, wa.

    Returns:
        np.ndarray: The E(z) function evaluated at z.
    """
    
    (
        t_H, H0, Ogamma0, Onu0,
        Om0, Ode0, massive_nu,
        N_eff, nu_y, n_massless_nu,
        N_eff_per_nu,
        w0, wa
    ) = args

    zp1 = 1 + z

    Or = Ogamma0 + (
                Ogamma0 * nu_relative_density(
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
        np.ndarray: The ODE at the given redshift.
    """

    cosmology_args = nb.carray(cosmology_args, (13,))
    z = nb.carray(z, (1,))

    t_H0 = cosmology_args[0]
    zp1 = 1+z
    Ez = E(z, cosmology_args)
    dz[0] = -1 * ( zp1 * Ez) / t_H0
z_ode_ptr = z_ode.address

def redshift_at_times(
    evaluation_times: np.ndarray,
    z0: float, t_H: float, H0: float, Ogamma0: float,
    Onu0: float, Om0: float, Ode0: float,
    massive_nu: float, N_eff: float, nu_y: float,
    n_massless_nu: float, N_eff_per_nu: float,
    w0: float, wa: float,
    rtol: float = 1e-8, atol: float = 1e-8,
    mxstep: int = 10000000

):
    
    z0 = np.array([z0])
    cosmology_args = np.array(
        [
            t_H, H0, Ogamma0, Onu0,
            Om0, Ode0, massive_nu,
            N_eff, nu_y, n_massless_nu,
            N_eff_per_nu,
            w0, wa
        ]
    )

    usol, success = lsoda(
        z_ode_ptr, z0, t_eval=evaluation_times, data=cosmology_args,
        rtol=rtol, atol=atol, mxstep=mxstep
    )

    return usol, success

@nb.njit
def lookback_time_integrand(
    z: float, cosmology_args: np.ndarray
) -> float:

    t_H0 = cosmology_args[0]
    zp1 = 1+z
    Ez = E(z, cosmology_args)
    integrand = t_H0 / (zp1 * Ez)

    return integrand

@nb.cfunc(quadpack_sig)
def lookback_time_integral(
    z, cosmology_args
):
    
    cosmology_args = nb.carray(cosmology_args, (13,))
    integral_value = lookback_time_integrand(z, cosmology_args)

    return integral_value
lookback_time_integral_ptr = lookback_time_integral.address

@nb.njit
def lookback_time(
    z_low: float, z_high: float, t_H: float, H0: float, 
    Ogamma0: float, Onu0: float, Om0: float, Ode0: float,
    massive_nu: float, N_eff: float, nu_y: float,
    n_massless_nu: float, N_eff_per_nu: float,
    w0: float, wa: float
) -> float:
    
    cosmology_args = np.array(
        [
            t_H, H0, Ogamma0, Onu0,
            Om0, Ode0, massive_nu,
            N_eff, nu_y, n_massless_nu,
            N_eff_per_nu,
            w0, wa
        ]
    )

    integral, _, _, _ = dqags(
        lookback_time_integral_ptr, z_low, z_high, cosmology_args
    )

    return integral

class Cosmology(Model):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        Ode0: float = 0.7,
        Tcmb0: float = 2.725, #update to planck value
        Neff: float = 3.046, #update to planck value
        m_nu_1: float = 0.0,
        m_nu_2: float = 0.0,
        m_nu_3: float = 0.06,
        Ob0: float = 0.04897,
    ):
        
        super().__init__()
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = Ode0
        self.Tcmb0 = Tcmb0
        self.Neff = Neff
        self.m_nu_1 = m_nu_1
        self.m_nu_2 = m_nu_2
        self.m_nu_3 = m_nu_3
        self.m_nu = np.array([m_nu_1, m_nu_2, m_nu_3])
        self.Ob0 = Ob0
        self.w0 = -1.0
        self.wa = 0.0

        self.cosmo = cosmo.LambdaCDM(
            H0=self.H0, Om0=self.Om0, Ode0=self.Ode0,
            Tcmb0=self.Tcmb0, Neff=self.Neff, m_nu=self.m_nu,
            Ob0=self.Ob0,
        )

        self.t_H = self.cosmo.hubble_time.to_value("Gyr")
        self.cosmo_args = np.array(
            [
                self.t_H, self.H0, self.cosmo._Ogamma0, 
                self.cosmo._Onu0, self.Om0, self.Ode0,
                self.cosmo._massivenu, self.cosmo._Neff,
                self.cosmo._nu_y[0], self.cosmo._nmasslessnu,
                self.cosmo._neff_per_nu, self.w0, self.wa
            ]
        )

    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def E(self, z: np.ndarray) -> np.ndarray:
        
        return E(z, self.cosmo_args)

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

        if z_at_value_kwargs['zmax'] > zmax:
            raise ValueError(
                'Upper limit for initial age to redshift ODE ' +
                'is above 1e10, something is wrong.'
            )

        warnings.filterwarnings('ignore')
        try:
            z0 = cosmo.z_at_value(
                self.cosmo.age, initial_time * Gyr,
                **z_at_value_kwargs
            ).value
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

    def redshift_at_times(self, evaluation_times: np.ndarray) -> np.ndarray:
        
        idx_sort = np.argsort(evaluation_times)
        idx_unsort = np.argsort(idx_sort)
        evaluation_times = evaluation_times[idx_sort]

        z0 = self.initial_redshift_value(
            evaluation_times[0],
        )

        usol, success = redshift_at_times(
            evaluation_times, z0, 
            *self.cosmo_args
        )
        usol = usol[idx_unsort, 0]

        return usol, success

class FlatLambdaCDM(Cosmology):

    def __init__(
        self,
        H0: float = 70.0,
        Om0: float = 0.3,
        Tcmb0: float = 2.725, #update to planck value
        Neff: float = 3.046, #update to planck value
        m_nu_1: float = 0.0,
        m_nu_2: float = 0.0,
        m_nu_3: float = 0.06,
        Ob0: float = 0.04897,
    ):
        
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=1-Om0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu_1=m_nu_1,
            m_nu_2=m_nu_2,
            m_nu_3=m_nu_3,
            Ob0=Ob0,
        )
        self.cosmo = cosmo.FlatLambdaCDM(
            H0=self.H0, Om0=self.Om0,
            Tcmb0=self.Tcmb0, Neff=self.Neff,
            m_nu=self.m_nu,
            Ob0=self.Ob0,
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
        m_nu_1: float = 0.0,
        m_nu_2: float = 0.0,
        m_nu_3: float = 0.06,
        Ob0: float = 0.04897,
        w0: float = -1.0,
    ):
        
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=1-Om0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu_1=m_nu_1,
            m_nu_2=m_nu_2,
            m_nu_3=m_nu_3,
            Ob0=Ob0,
        )
        self.w0 = w0
        self.cosmo = cosmo.FlatwCDM(
            H0=self.H0, Om0=self.Om0,
            Tcmb0=self.Tcmb0, Neff=self.Neff,
            m_nu=self.m_nu,
            Ob0=self.Ob0, w0=self.w0,
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
        m_nu_1: float = 0.0,
        m_nu_2: float = 0.0,
        m_nu_3: float = 0.06,
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
            m_nu_1=m_nu_1,
            m_nu_2=m_nu_2,
            m_nu_3=m_nu_3,
            Ob0=Ob0,
        )
        self.w0 = w0
        self.wa = wa
        self.cosmo = cosmo.Flatw0waCDM(
            H0=self.H0, Om0=self.Om0,
            Tcmb0=self.Tcmb0, Neff=self.Neff,
            m_nu=self.m_nu,
            Ob0=self.Ob0, w0=self.w0, wa=self.wa,
        )
    
    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distmod(z).value