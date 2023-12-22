import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import time
import pygtc
import blackjax
import arviz as az
import numpy as np
import numpyro as npy
import jax.numpy as jnp
import jax.random as random
import jax.scipy.stats as stats
import numpyro.distributions as dist
import numpyro.infer.util as infer_util

from jax.lax import cond
from datetime import date
from numpyro.handlers import seed, trace
from jax.experimental import host_callback
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_sample
from blackjax.util import run_inference_algorithm
from blackjax.mcmc.integrators import IntegratorState
from fastprogress.fastprogress import progress_bar
from numpyro.contrib.control_flow import scan


def initialize_model(rng_key, model, model_args=(), model_kwargs={}, num_chains=1):
    init_chain_keys = random.split(rng_key, num_chains)

    init_params, potential_fn, constrain_fn, _ = infer_util.initialize_model(
        rng_key=init_chain_keys,
        model=model,
        init_strategy=init_to_sample,
        model_args=model_args,
        model_kwargs=model_kwargs,
        dynamic_args=False,
    )
    init_positions = init_params.z

    logdensity_fn = lambda position: -potential_fn(position)

    init_state = jax.vmap(blackjax.mcmc.mclmc.init, (0, None, 0))(
        init_positions, logdensity_fn, init_chain_keys
    )

    init_tuning_state = IntegratorState(
        position={k: init_state.position[k][0] for k in init_state.position},
        momentum={k: init_state.momentum[k][0] for k in init_state.momentum},
        logdensity=init_state.logdensity[0],
        logdensity_grad={
            k: init_state.logdensity_grad[k][0] for k in init_state.logdensity_grad
        },
    )

    return logdensity_fn, constrain_fn, init_state, init_tuning_state


def tune_mclmc(
    rng_key, logdensity_fn, init_state, num_steps=500, target_varE=1e-4, verbose=False
):
    try:
        from blackjax.mcmc.integrators import noneuclidean_mclachlan as integrator
    except:
        from blackjax.mcmc.integrators import isokinetic_mclachlan as integrator

    kernel = blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=integrator,
    )

    if verbose:
        print(f"\nTuning MLMC for {num_steps} steps...")

    t0 = time.time()
    _, tuned_sampler_state = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=init_state,
        rng_key=rng_key,
        desired_energy_var=target_varE,
    )
    t1 = time.time()
    if verbose:
        print(f"Tuned L: {tuned_sampler_state.L}")
        print(f"Tuned step size: {tuned_sampler_state.step_size}")
        print(f"Tuning completed in {t1-t0:.2f} seconds.\n")

    tuned_mlmc = blackjax.mclmc(
        logdensity_fn=logdensity_fn,
        L=tuned_sampler_state.L,
        step_size=tuned_sampler_state.step_size,
    )

    return tuned_mlmc


def progress_bar_scan(num_samples, print_rate=None):
    "Progress bar for a JAX scan"
    progress_bars = {}

    if print_rate is None:
        if num_samples > 20:
            print_rate = int(num_samples / 20)
        else:
            print_rate = 1  # if you run the sampler for less than 20 iterations

    def _define_bar(arg, transform, device):
        progress_bars[0] = progress_bar(range(num_samples))
        progress_bars[0].update(0)

    def _update_bar(arg, transform, device):
        progress_bars[0].update_bar(arg)

    def _update_progress_bar(iter_num):
        "Updates progress bar of a JAX scan or loop"
        _ = cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(
                _define_bar, iter_num, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )

        _ = cond(
            # update every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0),
            lambda _: host_callback.id_tap(
                _update_bar, iter_num, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )

        _ = cond(
            # update by `remainder`
            iter_num == num_samples - 1,
            lambda _: host_callback.id_tap(
                _update_bar, num_samples, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )

    def _close_bar(arg, transform, device):
        progress_bars[0].on_iter_end()
        print()

    def close_bar(result, iter_num):
        return cond(
            iter_num == num_samples - 1,
            lambda _: host_callback.id_tap(
                _close_bar, None, result=result, tap_with_device=True
            ),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_bar(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan


def run_inference_algorithm(
    rng_key,
    initial_state,
    inference_algorithm,
    num_steps,
    num_chains=1,
    progress_bar=False,
):
    """Wrapper to run an inference algorithm.

    Note that this utility function does not work for Stochastic Gradient MCMC samplers
    like sghmc, as SG-MCMC samplers require additional control flow for batches of data
    to be passed in during each sample.

    Parameters
    ----------
    rng_key
        The random state used by JAX's random numbers generator.
    initial_state_or_position
        The initial state OR the initial position of the inference algorithm. If an initial position
        is passed in, the function will automatically convert it into an initial state.
    inference_algorithm
        One of blackjax's sampling algorithms or variational inference algorithms.
    num_steps
        Number of MCMC steps.
    progress_bar
        Whether to display a progress bar.
    transform
        A transformation of the trace of states to be returned. This is useful for
        computing determinstic variables, or returning a subset of the states.
        By default, the states are returned as is.

    Returns
    -------
    Tuple[State, State, Info]
        1. The final state of the inference algorithm.
        2. The trace of states of the inference algorithm (contains the MCMC samples).
        3. The trace of the info of the inference algorithm for diagnostics.
    """

    @jax.jit
    def _one_step(state, xs):
        _, rng_key = xs
        state, info = inference_algorithm.step(rng_key, state)
        return state, (state, info)

    if progress_bar:
        one_step = progress_bar_scan(num_steps)(_one_step)
    else:
        one_step = _one_step

    def sample_chain(rng_key, state):
        step_keys = random.split(rng_key, num_steps)
        xs = (jnp.arange(num_steps), step_keys)
        final_state, (state_history, info_history) = jax.lax.scan(one_step, state, xs)
        return final_state, state_history, info_history

    chain_keys = random.split(rng_key, num_chains)

    if progress_bar:
        print(f"\nBeginning sampling for {num_chains} chains and {num_steps} steps...")
    t0 = time.time()

    final_state, state_history, info_history = jax.pmap(sample_chain, (0, 0))(
        chain_keys, initial_state
    )
    final_state = jax.block_until_ready(final_state)
    state_history = jax.block_until_ready(state_history)
    info_history = jax.block_until_ready(info_history)

    t1 = time.time()
    if progress_bar:
        print(f"Sampling completed in {t1-t0:.2f} seconds.\n")

    return final_state, state_history, info_history


def flatten_parameter(parameter_arr):
    dims = parameter_arr.shape
    if len(dims) > 2:
        new_shape = (-1, *dims[2:])
    else:
        new_shape = (-1,)
    return jnp.reshape(parameter_arr, new_shape)


def unflatten_parameter(parameter_arr, n_chains=1):
    dims = parameter_arr.shape
    if len(dims) > 1:
        new_shape = (n_chains, -1, *dims[1:])
    else:
        new_shape = (
            n_chains,
            -1,
        )
    return jnp.reshape(parameter_arr, new_shape)


def constrain_parameters(state_history, constrain_fn, n_chains=1):
    if n_chains > 1:
        flattened_positions = jax.tree_map(flatten_parameter, state_history.position)
        constrained_positions = jax.vmap(constrain_fn)(flattened_positions)
        constrained_positions = jax.tree_map(
            lambda x: unflatten_parameter(x, n_chains), constrained_positions
        )
    else:
        constrained_positions = jax.vmap(constrain_fn)(state_history.position)

    return constrained_positions


def run_mclcm_sampler(
    rng_key,
    model,
    model_args=(),
    model_kwargs={},
    num_samples=1000,
    num_tuning_steps=500,
    num_chains=1,
    target_varE=1e-4,
    verbose=False,
):
    init_key, tuning_key, sampling_key = random.split(rng_key, 3)

    if verbose:
        print("\nInitializing model...")

    t0 = time.time()
    logdensity_fn, constrain_fn, init_state, init_tuning_state = initialize_model(
        rng_key=init_key,
        model=model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        num_chains=num_chains,
    )
    t1 = time.time()
    if verbose:
        print(f"Initialization completed in {t1-t0:.2f} seconds.\n")

    tuned_mlmc = tune_mclmc(
        rng_key=tuning_key,
        logdensity_fn=logdensity_fn,
        init_state=init_tuning_state,
        num_steps=num_tuning_steps,
        target_varE=target_varE,
        verbose=verbose,
    )

    final_state, state_history, info_history = run_inference_algorithm(
        rng_key=sampling_key,
        initial_state=init_state,
        inference_algorithm=tuned_mlmc,
        num_steps=num_samples,
        num_chains=num_chains,
        progress_bar=verbose,
    )

    if verbose:
        print("\nConstraining samples...")
    t0 = time.time()

    transformed_positions = constrain_parameters(
        state_history, constrain_fn, n_chains=num_chains
    )

    t1 = time.time()
    if verbose:
        print(f"Constraining completed in {t1-t0:.2f} seconds.\n")

    return final_state, state_history, info_history, transformed_positions


def arviz_from_states(positions, stats, chains=1, divergence_threshold=1e3):
    divergences = stats.energy_change > divergence_threshold
    posterior = az.from_dict(positions)
    sample_stats = az.convert_to_inference_data(
        {"diverging": divergences}, group="sample_stats"
    )
    trace = az.concat(posterior, sample_stats)

    return trace


def autocorrelation_time(x, num_chains=1):
    autocorr_at_lags = az.autocorr(x, axis=1)
    taus = np.zeros((autocorr_at_lags.shape[-1]))
    for i in range(num_chains):
        for j in range(autocorr_at_lags.shape[-1]):
            idx_neg = np.argmax(autocorr_at_lags[i, :, j] < 0)
            taus[j] += 2 * np.sum(autocorr_at_lags[i, :idx_neg, j]) - 1

    taus /= num_chains
    max_tau = np.max(taus)

    return max_tau
