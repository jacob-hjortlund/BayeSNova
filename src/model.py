import numpy as np
import numba as nb
import src.utils as utils
import scipy.stats as stats
import scipy.special as sp_special

from NumbaQuadpack import quadpack_sig, dqags
from astropy.cosmology import Planck18 as cosmo

class Model():

    def __init__(
        self, cfg: dict, observables: np.ndarray, covariances: np.ndarray
    ) -> None:

        self.mb = observables[:, 0]
        self.s = observables[:, 1]
        self.c = observables[:, 2]
        self.z = observables[:, 3]
        self.covs = covariances

        self.shared_par_names = cfg['shared_par_names']
        self.independent_par_names = cfg['independent_par_names']
        self.stretch_par_name = cfg['stretch_par_name']
        self.ratio_par_name = cfg['ratio_par_name']
        self.use_sigmoid = cfg['use_sigmoid']
        self.prior_bounds = cfg['prior_bounds']
        self.Ebv_integral_bounds = cfg.get('Ebv_bounds', None)
        self.Rb_bounds = cfg.get('Rb_bounds', None)

        if self.stretch_par_name in self.independent_par_names:
            self.stretch_independent = True
        else:
            self.stretch_independent = False

        if self.Ebv_integral_bounds is None:
            self.Ebv_quantiles = self.create_gamma_quantiles(cfg, 'Ebv')
        
        if self.Rb_bounds is None:
            self.Rb_quantiles = self.create_gamma_quantiles(cfg, 'Rb')

        self.__dict__.update(cfg['preset_values'])
    
    def create_gamma_quantiles(self, cfg: dict, par: str) -> np.ndarray:
        quantiles = utils.create_gamma_quantiles(
            cfg['prior_bounds']['gamma_' + par]['lower'],
            cfg['prior_bounds']['gamma_' + par]['upper'],
            cfg['resolution_g' + par], cfg['cdf_limit_g' + par]
        )
        return quantiles

    def log_prior(self, par_dict: dict) -> float:
        
        value = 0.
        stretch_1_par = self.stretch_par_name + "_1"
        stretch_2_par = self.stretch_par_name + "_2"

        for value_key in self.par_dict.keys():
            
            # TODO: Remove 3-deep conditionals bleeeeh
            bounds_key = ""
            is_independent_stretch = (
                (
                    value_key == stretch_1_par or value_key == stretch_2_par
                ) and self.stretch_independent
            )
            is_not_ratio_par_name = value_key != self.ratio_par_name

            if is_independent_stretch:
                if not self.__dict__[stretch_1_par] < self.__dict__[stretch_2_par]:
                    value += -np.inf
                    break
            
            if is_not_ratio_par_name:
                bounds_key = "_".join(value_key.split("_")[:-1])
            else:
                bounds_key = value_key
            
            is_in_priors = bounds_key in self.prior_bounds.keys()
            if is_in_priors:
                value += utils.uniform(
                    par_dict[value_key], **self.prior_bounds[bounds_key]
                )

            if np.isinf(value):
                break

        return value
    
    def population_covariances(self) -> np.ndarray:

        cov = np.tile(self.covs, (2, 1, 1, 1))
        disp_v_pec = 200. # km / s
        c = 300000. # km / s
        
        cov[:,:,0,0] += np.array([
            [self.sig_int_1**2 + self.alpha_1**2 * self.sig_s_1**2 + self.beta_1**2 * self.sig_c_1**2],
            [self.sig_int_2**2 + self.alpha_2**2 * self.sig_s_2**2 + self.beta_2**2 * self.sig_c_2**2]
        ])
        cov[:,:,0,0] += np.tile(
            self.z**(-2) * (
                (5. / np.log(10.))**2
                * (disp_v_pec / c)**2
            ), (2, 1)
        )

        cov[:,:,1,1] = np.array([
            [self.sig_s_1**2], [self.sig_s_2**2]
        ])
        cov[:,:,2,2] = np.array([
            [self.sig_c_1**2], [self.sig_c_2**2]
        ])
        cov[:,:,0,1] = np.array([
            [self.alpha_1 * self.sig_s_1**2], [self.alpha_2 * self.sig_s_2**2]
        ])
        cov[:,:,0,2] = np.array([
            [self.beta_1 * self.sig_c_1**2], [self.beta_2 * self.sig_c_2**2]
        ])
        cov[:,:,1,0] = cov[:,:,0,1]
        cov[:,:,2,0] = cov[:,:,0,2]

        return cov[0], cov[1]

    def population_residuals(self) -> np.ndarray:
        pass
    
    def Ebv_prior_convolutions(self) -> float:
        pass

    def Rb_Ebv_prior_convolutions(self) -> float:
        pass

    def log_likelihood(self) -> float:
        pass

    def __call__(self, theta: np.ndarray) -> float:

        param_dict = utils.theta_to_dict(
            theta=theta, shared_par_names=self.shared_par_names,
            independent_par_names=self.independent_par_names,
            ratio_par_name=self.ratio_par_name
        )

        log_prior = self.log_prior()
        if np.isinf(log_prior):
            return log_prior

        self.__dict__.update(param_dict)
        
        if self.use_sigmoid:
            self.__dict__.update(
                utils.apply_sigmoid(
                    self.__dict__, sigmoid_cfg=self.sigmoid_cfg,
                    independent_par_names=self.independent_par_names,
                    ratio_par_name=self.ratio_par_name
                )
            )
        
        log_likelihood = self.log_likelihood()

        return log_likelihood

        pass

    