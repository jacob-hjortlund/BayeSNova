import numpy as np

from bayesnova.progenitors import SNeProgenitors
from bayesnova.base import Gaussian, Mixture, Weighting

class ConstantWeighting(Weighting):

    def __init__(
        self,
        weight: float
    ) -> None:
        super().__init__()
        self.weight = weight
    
    def calculate_weight(
        self,
        redshift: float,
        **kwargs
    ):
        return np.ones_like(redshift) * self.weight

class LogisticLinearWeighting(Weighting):

    def __init__(
        self,
        scale: float = 1.0,
        offset: float = 0.0,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset

    def logistic(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def inverse_logistic(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        return np.log(y / (1.0 - y))

    def calculate_weight(
        self,
        weights: np.ndarray,
        **kwargs
    ):
        
        unbound_weights = self.inverse_logistic(weights)
        linear_unbound_weights = self.scale * unbound_weights + self.offset
        linear_weights = self.logistic(linear_unbound_weights)
        
        return linear_weights

class RedshiftDependentWeighting(Weighting):

    def __init__(
        self,
        progenitor_model: SNeProgenitors,
    ):
        super().__init__()
        self.progenitor_model = progenitor_model
    
    def calculate_weight(
        self,
        redshift: float,
        **kwargs
    ):
        rates = self.progenitor_model.volumetric_rates(redshift)
        rates_are_finite = np.all(np.isfinite(rates))

        if rates_are_finite:
            total_rate = rates[:,0]
            delayed_rate = rates[:,2]
            weight = delayed_rate / total_rate
        else:
            weight = np.ones_like(redshift) * -np.inf

        return weight

class TwoPopulationMixture(Mixture):

    def __init__(
        self,
        population_models: list[Gaussian],
        weighting_model: Gaussian,
    ):
        super().__init__(
            population_models=population_models,
            weighting_model=weighting_model,
        )
        self.n_populations = len(population_models)
        if self.n_populations != 2:
            raise ValueError(f"Expected 2 population models, got {self.n_populations}.")
    
    def log_likelihood(
        self,
        **kwargs,
    ) -> np.ndarray:
        
        log_likelihoods = []

        for i, population_model in enumerate(self.population_models):
            log_likelihoods_i = population_model.log_likelihood(
                **kwargs,
            )
            log_likelihoods.append(log_likelihoods_i)
        
        population_1_weight = self.weighting_model.calculate_weight(
            **kwargs
        )
        population_2_weight = 1.0 - population_1_weight

        weight_is_finite = np.all(np.isfinite(population_1_weight))

        if weight_is_finite:
            log_likelihoods = np.logaddexp(
                np.log(population_1_weight) + log_likelihoods[0],
                np.log(population_2_weight) + log_likelihoods[1],
            )
            log_likelihood = np.sum(log_likelihoods)

        else:
            log_likelihood = -np.inf

        return log_likelihood