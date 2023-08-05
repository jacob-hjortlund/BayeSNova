import numpy as np

class Model():

    def __init__(self) -> None:
        pass

    def log_likelihood(self, **kwargs):
        
        raise NotImplementedError

class Gaussian(Model):

    def __init__(self) -> None:
        super().__init__()

    def residual(self, **kwargs):
        
        raise NotImplementedError
    
    def covariance(self, **kwargs):
        
        raise NotImplementedError

class UnivariateGaussian(Gaussian):

    def __init__(
        self,
        mu: float,
        sigma: float
    ) -> None:
        super().__init__()

        self.mu = mu
        self.sigma = sigma

    def log_likelihood(
        self,
        observations: np.ndarray,
        variance: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        
        var = self.sigma**2 + variance
        norm = -0.5 * np.log(2 * np.pi * var)
        idx_inf = np.isinf(norm)
        norm[idx_inf] = -1e99

        return norm - 0.5 * (observations - self.mu)**2 / var
    
class Weighting(Model):

    def __init__(self) -> None:
        super().__init__()

    def calculate_weight(self, **kwargs):
        
        raise NotImplementedError

class Mixture(Model):

    def __init__(
        self,
        population_models: list[Gaussian],
        weighting_model: Weighting
    ) -> None:
        super().__init__()

        self.population_models = population_models
        self.weighting_model = weighting_model
    
    def log_likelihood(
        self,
        redshift: float,
        **kwargs
    ) -> np.ndarray:
        raise NotImplementedError
        