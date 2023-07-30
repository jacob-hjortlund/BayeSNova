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
        