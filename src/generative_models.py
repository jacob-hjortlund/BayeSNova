import numpy as np
import scipy.stats as stats
import astropy.cosmology as apy_cosmo
import src.preprocessing as prep
import src.cosmology_utils as cosmo_utils

from astropy.units import Gyr

class SNModel():

    def __init__(
        self, model_cfg: dict
    ):
        self.cfg = model_cfg
        self.init_cosmology(
            H0=model_cfg["cosmology"]["H0"],
            Om0=model_cfg["cosmology"]["Om0"],
            w0=model_cfg["cosmology"]["w0"],
            wa=model_cfg["cosmology"]["wa"]
        )

    def init_cosmology(
        self, H0: float, Om0: float, w0: float, wa: float
    ):
        self.cosmo = apy_cosmo.Flatw0waCDM(
            H0=H0, Om0=Om0, w0=w0, wa=wa
        )

    def sample_reddening(
        self, n_sn: int, gamma_Ebv: float, tau_Ebv: float
    ):
        
        gamma = stats.gamma(a=gamma_Ebv, scale=tau_Ebv)
        samples = gamma.rvs(size=n_sn)
        return samples
    
    def population_covarainces(
        self,
        z: np.ndarray, obs_covariance: np.ndarray,
        alpha_1: float, alpha_2: float,
        beta_1: float, beta_2: float,
        sig_int_1: float, sig_int_2: float,
        sig_s_1: float, sig_s_2: float,
        sig_c_1: float, sig_c_2: float,
    ) -> np.ndarray:

        n_sn = len(z)
        cov = np.tile(obs_covariance, (2, n_sn, 1, 1))
        disp_v_pec = 200. # km / s
        c = 300000. # km / s
        
        cov[:,:,0,0] += np.array([
            [sig_int_1**2 + alpha_1**2 * sig_s_1**2 + beta_1**2 * sig_c_1**2],
            [sig_int_2**2 + alpha_2**2 * sig_s_2**2 + beta_2**2 * sig_c_2**2]
        ])
        cov[:,:,0,0] += np.tile(
            z**(-2) * (
                (5. / np.log(10.))**2
                * (disp_v_pec / c)**2
            ), (2, 1)
        )

        cov[:,:,1,1] += np.array([
            [sig_s_1**2], [sig_s_2**2]
        ])
        cov[:,:,2,2] += np.array([
            [sig_c_1**2], [sig_c_2**2]
        ])
        cov[:,:,0,1] += np.array([
            [alpha_1 * sig_s_1**2], [alpha_2 * sig_s_2**2]
        ])
        cov[:,:,0,2] += np.array([
            [beta_1 * sig_c_1**2], [beta_2 * sig_c_2**2]
        ])
        cov[:,:,1,0] = cov[:,:,0,1]
        cov[:,:,2,0] = cov[:,:,0,2]

        return cov

    def population_mean(
        self, z: np.ndarray,
        Mb: float, s: float,
        c_int: float, alpha: float,
        beta: float, Rb: float,
        Ebv: np.ndarray
    ):

        n_sn = len(z)
        means = np.zeros((n_sn, 3))
        means[:,0] = (
            Mb + alpha * s + beta * c_int + self.cosmo.distmod(z).value + Rb * Ebv
        )
        means[:,1] = s
        means[:,2] = c_int + Ebv

        return means
        pass

    def volumetric_sn_rates(
        self, z: np.ndarray,
        eta_prompt: float, eta_delayed: float
    ):

        dtd_t0 = self.cfg['dtd_cfg']['t0']
        dtd_t1 = self.cfg['dtd_cfg']['t1']

        H0 = self.cosmo.H0.value
        H0_gyrm1 = self.cosmo.H0.to(1/Gyr).value
        Om0 = self.cosmo.Om0
        w0 = self.cosmo.w0
        wa = self.cosmo.wa
        cosmo_args = (H0_gyrm1,Om0,1.-Om0,w0, wa)

        convolution_time_limits = cosmo_utils.convolution_limits(
            self.cosmo, z, dtd_t0, dtd_t1
        )
        minimum_convolution_time = np.min(convolution_time_limits)        
        z0 = apy_cosmo.z_at_value(
            self.cosmo.age, minimum_convolution_time * Gyr,
            method='Bounded'
        )
        _, zs, _ = cosmo_utils.redshift_at_times(
            convolution_time_limits, minimum_convolution_time, z0, cosmo_args
        )
        integral_limits = np.array(zs.tolist(), dtype=np.float64)
        sn_rates = cosmo_utils.volumetric_rates(
            z, integral_limits, H0, Om0,
            w0, wa, eta_prompt, eta_delayed, zinf=20.
        )

        return sn_rates

    def __call__(
        self, z: np.ndarray, obs_covariance: np.ndarray
    ):
        
        n_sn = len(z)
        
        Ebv = self.sample_reddening(
            n_sn=n_sn,
            gamma_Ebv=self.cfg['reddening']['gamma_Ebv'],
            tau_Ebv=self.cfg['reddening']['tau_Ebv']
        )

        pop_1_means = self.population_mean(
            z=z, Mb=self.cfg['pop_1']['Mb'],
            s=self.cfg['pop_1']['s'], c_int=self.cfg['pop_1']['c_int'],
            alpha=self.cfg['pop_1']['alpha'], beta=self.cfg['pop_1']['beta'],
            Rb=self.cfg['pop_1']['Rb'], Ebv=Ebv
        )

        pop_2_means = self.population_mean(
            z=z, Mb=self.cfg['pop_2']['Mb'],
            s=self.cfg['pop_2']['s'], c_int=self.cfg['pop_2']['c_int'],
            alpha=self.cfg['pop_2']['alpha'], beta=self.cfg['pop_2']['beta'],
            Rb=self.cfg['pop_2']['Rb'], Ebv=Ebv
        )
        means = np.column_stack((pop_1_means, pop_2_means))

        covs = self.population_covarainces(
            z=z, obs_covariance=obs_covariance,
            alpha_1=self.cfg['pop_1']['alpha'],
            alpha_2=self.cfg['pop_2']['alpha'],
            beta_1=self.cfg['pop_1']['beta'],
            beta_2=self.cfg['pop_2']['beta'],
            sig_int_1=self.cfg['pop_1']['sig_int'],
            sig_int_2=self.cfg['pop_2']['sig_int'],
            sig_s_1=self.cfg['pop_1']['sig_s'],
            sig_s_2=self.cfg['pop_2']['sig_s'],
            sig_c_1=self.cfg['pop_1']['sig_c'],
            sig_c_2=self.cfg['pop_2']['sig_c']
        )
        covs = covs.T
        
        eta_prompt = self.cfg['dtd_cfg']['eta_prompt']
        eta_delayed = self.cfg['dtd_cfg']['eta_delayed']
        sn_rates = self.volumetric_sn_rates(
            z=z, eta_prompt=eta_prompt, eta_delayed=eta_delayed
        )

        pop_1_probability = sn_rates[:,-1] / sn_rates[:, 0]
        pop_2_probability = 1. - pop_1_probability
        sample_idx = stats.binom.rvs(n=1, p=pop_2_probability)

        sample_means = means[sample_idx]
        sample_covs = covs[sample_idx]

        pass