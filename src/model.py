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
        self.ratio_par_name = cfg['ratio_par_name']
        self.use_sigmoid = cfg['use_sigmoid']
        self.prior_bounds = cfg['prior_bounds']
        self.Ebv_integral_bounds = cfg.get('Ebv_bounds', None)
        self.Rb_bounds = cfg.get('Rb_bounds', None)
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

    def log_prior() -> float:
        pass
    
    def population_covariances(self) -> np.ndarray:
        pass

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
        self.__dict__.update(param_dict)

        log_prior = self.log_prior()
        if np.isinf(log_prior):
            return log_prior
        
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

    