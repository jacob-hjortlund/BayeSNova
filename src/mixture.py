import numpy as np

from base import Gaussian, Mixture, Weighting

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
    
    def _log_likelihood(
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

        log_likelihoods = np.logaddexp(
            np.log(population_1_weight) + log_likelihoods[:, 0],
            np.log(population_2_weight) + log_likelihoods[:, 1],
        )
        log_likelihood = np.sum(log_likelihoods)

        return log_likelihood