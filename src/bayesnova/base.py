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


def get_model_paths(model):
    strings = []
    if isinstance(model, zdx.Base):
        paths = jax.tree_util.tree_leaves_with_path(model)

        for i in range(len(paths)):
            path = paths[i][0]
            string = jax.tree_util.keystr(path)

            new_string = re.sub(r"\.(\w+)\['(\w+)'\]\.", r"\2.", string)
            new_string = re.sub(r"^\.", "", new_string)  # remove leading "."
            new_string = re.sub(
                r"\[.*\]", "", new_string
            )  # remove flat indices from dists
            strings = strings + [new_string]

    return strings


class Constant(zdx.Base):
    name: str
    value: Union[Array, zdx.Base]

    def __init__(
        self,
        value: Array,
        name: str = "constant",
    ):
        self.name = name
        self.value = jnp.asarray(value, dtype=jnp.float64)

    def __call__(self, *args, **kwargs):
        return npy.deterministic(self.name, self.value)


class Base(zdx.Base):
    name: str
    constraints: list
    constraint_factor: float

    def __init__(self, name: str = "base", constraint_factor: float = -500.0):
        self.name = name
        self.constraints = []
        self.constraint_factor = constraint_factor

    def _constant_to_lambda(self, arg, name="constant"):
        if not isinstance(arg, list):
            arg = [arg]

        nodes = lambda x: tuple(
            xi
            for xi, yi in zip(x, arg)
            if (
                isinstance(yi, jnp.ndarray)
                or isinstance(yi, float)
                or isinstance(yi, int)
            )
        )

        replace_fn = lambda leaf: Constant(leaf, name=name)
        return zdx.eqx.tree_at(nodes, arg, replace_fn=replace_fn)[0]

    def _rename_submodel(self, submodel, modifiers: Union[str, list[str]] = []):
        if isinstance(modifiers, str):
            modifiers = [modifiers]

        model_paths = get_model_paths(submodel)
        for path in model_paths:
            if "name" in path:
                old_name = submodel.get(path)
                split_path = path.split(".")
                path_list = [self.name] + modifiers + split_path[:-1] + [old_name]
                new_name = "_".join(path_list)
                submodel = submodel.set(path, new_name)

        return submodel

    def add_constraint(
        self, parameters: Union[zdx.Base, list[zdx.Base]], constraint_fn: Callable
    ):
        """Adds a constraint to the distribution.

        Args:
            parameters (Union[str, list[str]]): The parameter(s) to add the constraint to.
            constraint_fn (Callable): A function that takes the parameters as arguments
                and returns a constraint.
        """

        if isinstance(parameters, zdx.Base):
            parameters = [parameters]

        # self.constraints = constraints + [(parameters, constraint_fn)]
        self.constraints.append((parameters, constraint_fn))

    def apply_constraints(self, stochastic_fn: Callable, *args, **kwargs):
        """Applies the constraints to the model.

        Args:
            stochastic_fn (Callable): A function that returns a stochastic value.
            *args: Arguments to pass to the stochastic function.
            **kwargs: Keyword arguments to pass to the stochastic function.
        """
        trace = npy.handlers.trace(stochastic_fn).get_trace(*args, **kwargs)
        constraints = jnp.asarray(
            [
                constraint_fn(
                    *[trace[parameter.name]["value"] for parameter in parameters]
                )
                for parameters, constraint_fn in self.constraints
            ]
        )
        constraint = jnp.all(constraints)
        npy.factor(
            self.name + "_constraints",
            jnp.where(constraint, 0.0, self.constraint_factor),
        )
        sample = trace[self.name]["value"]
        return sample


class ModelCollection(zdx.Base):
    """A collection of arbitrary Zodiax models that can be used as a single model."""

    models: dict

    def __init__(self, model_list: list, names: Union[list[str], str] = "model"):
        """
        Args:
            model_list (list): A list of Zodiax models.
            names (list[str], str): A list of names for the models. If a string is
                provided, the models will be named as f"{names}_{i}" for i in
                range(len(model_list)).
        """

        if isinstance(names, str):
            names = [f"{names}_{i}" for i in range(len(model_list))]
        elif len(names) != len(model_list):
            raise ValueError("Length of names must be equal to length of model_list")

        models = {}
        for i, model in enumerate(model_list):
            name = names[i]
            model_paths = get_model_paths(model)
            for path in model_paths:
                if "name" in path:
                    old_name = model.get(path)
                    split_path = path.split(".")
                    path_list = [name] + split_path[:-1] + [old_name]
                    new_name = ".".join(path_list)
                    model = model.set(path, new_name)
            models[names[i]] = model
        self.models = models

    def __getattr__(self, key):
        """Allows us to access the individual models by their dictionary key"""
        if key in self.models.keys():
            return self.models[key]
        else:
            raise AttributeError(f"{key} not in {self.models.keys()}")

    def __call__(self, *args, **kwargs):
        return [model.dist(*args, **kwargs) for model in self.models.values()]


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
