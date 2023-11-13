import re
import jax
import zodiax as zdx
import numpyro as npy
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import numpyro.distributions as dist

from jax import Array
from jax.lax import cond
from typing import Union, Any, Callable
from bayesnova.base import Base


class Distribution(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dist(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        obs_key = self.name + "_obs"
        value = npy.sample(
            self.name,
            self.dist(*args, **kwargs),
            obs=kwargs.get(obs_key),
        )
        return value

    def __call__(self, *args, **kwargs):
        sample = self.apply_constraints(self.sample, *args, **kwargs)
        return sample


class Uniform(Distribution):
    low: Union[Array, zdx.Base]
    high: Union[Array, zdx.Base]

    def __init__(
        self,
        low: Union[Array, zdx.Base] = jnp.array(0.0, dtype=jnp.float64),
        high: Union[Array, zdx.Base] = jnp.array(1.0, dtype=jnp.float64),
        name: str = "uniform",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.low = self._rename_submodel(self._constant_to_lambda(low, name="low"))
        self.high = self._rename_submodel(self._constant_to_lambda(high, name="high"))

    def dist(self, *args, **kwargs):
        low = self.low(*args, **kwargs)
        high = self.high(*args, **kwargs)
        return dist.Uniform(low, high)


class Normal(Distribution):
    mean: Union[Array, zdx.Base]
    std: Union[Array, zdx.Base]

    def __init__(
        self,
        mean: Union[Array, zdx.Base] = jnp.array(0.0, dtype=jnp.float64),
        std: Union[Array, zdx.Base] = jnp.array(1.0, dtype=jnp.float64),
        name: str = "normal",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.mean = self._rename_submodel(self._constant_to_lambda(mean, name="mean"))
        self.std = self._rename_submodel(self._constant_to_lambda(std, name="std"))

    def dist(self, *args, **kwargs):
        mean = self.mean(*args, **kwargs)
        std = self.std(*args, **kwargs)
        return dist.Normal(mean, std)


class MultivariateNormal(Distribution):
    mean: Union[Array, zdx.Base]
    cov: Union[Array, zdx.Base]

    def __init__(
        self,
        mean: Union[Array, zdx.Base] = jnp.array(0.0, dtype=jnp.float64),
        cov: Union[Array, zdx.Base] = jnp.array(1.0, dtype=jnp.float64),
        name: str = "multivariate_normal",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.mean = self._rename_submodel(self._constant_to_lambda(mean, name="mean"))
        self.cov = self._rename_submodel(self._constant_to_lambda(cov, name="cov"))

    def dist(self, *args, **kwargs):
        mean = self.mean(*args, **kwargs)
        cov = self.cov(*args, **kwargs)
        return dist.MultivariateNormal(mean, cov)


class Gamma(Distribution):
    concentration: Union[Array, zdx.Base]
    rate: Union[Array, zdx.Base]

    def __init__(
        self,
        concentration: Union[Array, zdx.Base] = jnp.array(1.0, dtype=jnp.float64),
        rate: Union[Array, zdx.Base] = jnp.array(1.0, dtype=jnp.float64),
        name: str = "gamma",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.concentration = self._rename_submodel(
            self._constant_to_lambda(concentration, name="concentration")
        )
        self.rate = self._rename_submodel(self._constant_to_lambda(rate, name="rate"))

    def dist(self, *args, **kwargs):
        concentration = self.concentration(*args, **kwargs)
        rate = self.rate(*args, **kwargs)
        return dist.Gamma(concentration, rate)


class TwoComponentMixture(Distribution):
    models: dict
    mixture_weight: Union[Array, zdx.Base]

    def __init__(
        self,
        models: list[zdx.Base],
        mixture_weight: Union[Array, zdx.Base],
        name: str = "mixture_model",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.models = {
            f"{self.name}_{i}": self._rename_submodel(model)
            for i, model in enumerate(models)
        }
        self.mixture_weight = self._rename_submodel(
            self._constant_to_lambda(mixture_weight, name="mixture_weight")
        )

    def __getattr__(self, key):
        """Allows us to access the individual models by their dictionary key"""
        if key in self.models.keys():
            return self.models[key]
        else:
            raise AttributeError(f"{key} not in {self.models.keys()}")

    def model_dists(self, *args, **kwargs):
        return [model.dist(*args, **kwargs) for model in self.models.values()]

    def dist(self, model_dists: list, mixture_weight: Array, *args, **kwargs):
        mixture_weights = jnp.array([mixture_weight, 1.0 - mixture_weight])
        model = dist.MixtureGeneral(
            dist.Categorical(probs=mixture_weights),
            model_dists,
        )

        return model

    def log_membership_ratio(
        self,
        samples: Array,
        model_dists: list,
        mixture_weight: Array,
        name: str = "log_membership_ratio",
        *args,
        **kwargs,
    ):
        log_prob_1 = model_dists[0].log_prob(samples) * jnp.log(mixture_weight)
        log_prob_2 = model_dists[1].log_prob(samples) * jnp.log((1.0 - mixture_weight))

        log_membership_ratio = log_prob_1 - log_prob_2
        npy.deterministic(name, log_membership_ratio)

    def sample(self, *args, **kwargs):
        model_dists = self.model_dists(*args, **kwargs)
        mixture_weight = self.mixture_weight(*args, **kwargs)

        obs_key = self.name + "_obs"
        value = npy.sample(
            self.name,
            self.dist(
                model_dists=model_dists, mixture_weight=mixture_weight, *args, **kwargs
            ),
            obs=kwargs.get(obs_key),
        )

        self.log_membership_ratio(
            value,
            model_dists=model_dists,
            mixture_weight=mixture_weight,
            name=self.name + "_log_membership_ratio",
        )

        return value
