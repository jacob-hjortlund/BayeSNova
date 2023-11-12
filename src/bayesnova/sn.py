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


class SaltSNStandardization(Base):
    M: Union[Array, zdx.Base]
    alpha: Union[Array, zdx.Base]
    beta: Union[Array, zdx.Base]
    X_1_int: Union[Array, zdx.Base]
    c_int: Union[Array, zdx.Base]
    R_B: Union[Array, zdx.Base]
    E_BV: Union[Array, zdx.Base]

    def __init__(
        self,
        M: Union[Array, zdx.Base],
        mu: Union[Array, zdx.Base],
        alpha: Union[Array, zdx.Base],
        beta: Union[Array, zdx.Base],
        X_1_int: Union[Array, zdx.Base],
        c_int: Union[Array, zdx.Base],
        R_B: Union[Array, zdx.Base],
        E_BV: Union[Array, zdx.Base],
        R_B_lower_limit: Union[Array, zdx.Base] = jnp.array(0.0, dtype=jnp.float64),
        name: str = "sn",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.alpha = self._rename_submodel(
            self._constant_to_lambda(alpha, name="alpha")
        )
        self.beta = self._rename_submodel(self._constant_to_lambda(beta, name="beta"))

        self.M = self._rename_submodel(self._constant_to_lambda(M, name="M"))
        self.mu = self._rename_submodel(self._constant_to_lambda(mu, name="mu"))
        self.X_1_int = self._rename_submodel(
            self._constant_to_lambda(X_1_int, name="X_1_int")
        )
        self.c_int = self._rename_submodel(
            self._constant_to_lambda(c_int, name="c_int")
        )

        self.R_B = self._rename_submodel(self._constant_to_lambda(R_B, name="R_B"))
        self.E_BV = self._rename_submodel(self._constant_to_lambda(E_BV, name="E_BV"))

        self.add_constraint(
            self.R_B,
            lambda R_B: R_B > R_B_lower_limit,
        )

    def design_matrix(self, *args, **kwargs):
        design_matrix = jnp.array(
            [
                [
                    1.0,
                    1.0,
                    self.alpha(*args, **kwargs),
                    self.beta(*args, **kwargs),
                    self.R_B(*args, **kwargs),
                ],
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )

        return design_matrix

    def coefficients(self, *args, **kwargs):
        coefficients = jnp.array(
            [
                self.M(*args, **kwargs),
                self.mu(*args, **kwargs),
                self.X_1_int(*args, **kwargs),
                self.c_int(*args, **kwargs),
                self.E_BV(*args, **kwargs),
            ],
            dtype=jnp.float64,
        )

        return coefficients

    def model(self, *args, **kwargs):
        design_matrix = self.design_matrix(*args, **kwargs)
        coefficients = self.coefficients(*args, **kwargs)
        model = design_matrix @ coefficients
        return model

    def __call__(self, *args, **kwargs):
        value = self.apply_constraints(self.model, *args, **kwargs)
        return value
