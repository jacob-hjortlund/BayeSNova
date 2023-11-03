import numpy as np


class Model:
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
    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__()

        self.mu = mu
        self.sigma = sigma

    def log_likelihood(
        self, observations: np.ndarray, variance: np.ndarray, **kwargs
    ) -> np.ndarray:
        observations = np.asarray(observations)
        variance = np.asarray(variance)

        idx_observed = variance < 1e150
        output = np.ones(len(observations)) * -354.891356446692  # log of largest float

        var = self.sigma**2 + variance[idx_observed]
        norm = -0.5 * np.log(2 * np.pi * var)
        chisq = (observations[idx_observed] - self.mu) ** 2 / var
        output[idx_observed] = norm - 0.5 * chisq

        return output


class Weighting(Model):
    def __init__(self) -> None:
        super().__init__()

    def calculate_weight(self, **kwargs):
        raise NotImplementedError


class Mixture(Model):
    def __init__(
        self, population_models: list[Gaussian], weighting_model: Weighting
    ) -> None:
        super().__init__()

        self.population_models = population_models
        self.weighting_model = weighting_model

    def log_likelihood(self, redshift: float, **kwargs) -> np.ndarray:
        raise NotImplementedError
