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


class LinearModel(Base):
    """A 1D linear model with slope and intercept."""

    slope: Union[Array, zdx.Base]
    intercept: Union[Array, zdx.Base]

    def __init__(
        self,
        slope: Union[Array, zdx.Base] = jnp.array(1.0, dtype=jnp.float64),
        intercept: Union[Array, zdx.Base] = jnp.array(0.0, dtype=jnp.float64),
        name: str = "linear_model",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.slope = self._rename_submodel(
            self._constant_to_lambda(slope, name="slope")
        )
        self.intercept = self._rename_submodel(
            self._constant_to_lambda(intercept, name="intercept")
        )

    def model(self, *args, **kwargs):
        slope = self.slope(*args, **kwargs)
        intercept = self.intercept(*args, **kwargs)
        input_key = self.name + "_input"
        value = npy.deterministic(
            self.name,
            slope * kwargs.get(input_key) + intercept,
        )

        return value

    def __call__(self, *args, **kwargs):
        sample = self.apply_constraints(self.model, *args, **kwargs)
        return sample
