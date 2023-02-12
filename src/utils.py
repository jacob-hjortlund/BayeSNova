import sys
import tqdm
import yaml
import time
import omegaconf
import numpy as np
import numba as nb
import emcee as em
import schwimmbad as swbd
import deepdiff.diff as diff
import scipy.stats as sp_stats
import scipy.optimize as sp_opt
import scipy.special as sp_special

from mpi4py import MPI

NULL_VALUE = -9999.

class _FunctionWrapper:
    """
    Object to wrap user's function, allowing picklability
    """
    def __init__(self, f, args=()):
        self.f = f
        if not isinstance(args, tuple):
            args = (args,)
        self.args = [] if args is None else args

    def __call__(self, x):
        return self.f(x, *self.args)

class PoolWrapper():

    def __init__(self, pool_type: str) -> None:
        self.pool_type = str(pool_type or "").upper()
        self.pool = None
        self.is_mpi = False
        self.check_mpi()
        self.set_pool()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pool != None:
            self.pool.close()

    def check_mpi(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        if size > 1 and self.pool_type != "MPI":
            raise Exception # only use mpirun with MPI pool
        elif size == 1 and self.pool_type == "MPI":
            raise Exception #cant create mpi pool with only 1 procces
        elif self.pool_type == "MPI":
            self.is_mpi = True
        else:
            pass

    def set_pool(self):
        if self.pool_type == "MPI":
            self.pool = swbd.MPIPool()
        if self.pool_type == "MP":
            raise NotImplementedError("MP/Multiprocessing support not yet implemented")
    
    def check_if_master(self):
        if not self.is_mpi:
            raise Exception("Check if master only applicable for MPI pools.")
        if not self.pool.is_master():
            self.pool.wait()
            sys.exit(0)

def estimate_mmap(samples):


    mean_val = np.mean(samples)
    t0 = time.time()
    kernel = sp_stats.gaussian_kde(samples)
    t1 = time.time()

    t2 = time.time()
    output = sp_opt.minimize(
        lambda x: -kernel(x),
        mean_val,
        method='Nelder-Mead',
        options={'disp': False}
    ).x[0]
    t3 = time.time()

    print(f"\nKDE took {t1-t0} seconds to run and {t3-t2} seconds to optimize.\n")


    return output
    
def create_task_name(
    cfg: omegaconf.DictConfig, default_path: str ='./configs/config.yaml'
) -> str:

    cfg = yaml.safe_load(
        omegaconf.OmegaConf.to_yaml(cfg)
    )
    with open(default_path, "r") as f:
        default_cfg = yaml.safe_load(f)

    diff_dict = diff.DeepDiff(default_cfg, cfg)
    if not 'values_changed' in diff_dict.keys():
        return 'default_cfg'

    changes = []
    for setting in diff_dict['values_changed'].keys():
        if (
            "independent_par_name" in setting or
            "clearml_cfg" in setting
        ):
            continue
        setting_str = '['+"[".join(setting.split('[')[2:])
        new_value = str(diff_dict['values_changed'][setting]['new_value'])
        if "shared_par_name" in setting:
            changes.append(new_value)
        else:
            changes.append(
                setting_str + '-' + new_value
            )
    run_name = '_'.join(changes)
    run_name = run_name.replace("'", "")
    run_name = run_name.replace("][", "_")
    run_name = run_name.replace("[", "")
    run_name = run_name.replace("]", "")

    return run_name

def sigmoid(
    value: float, shift: float = 0.,
    scale: float = 1., divisor: int = 1
):

    exponent = (- scale * (value - shift)) / divisor
    denom = 1. + divisor * np.exp(exponent)

    return 1. / denom

def uniform(value: float, lower: float = -np.inf, upper: float = np.inf):
    if type(value) == type(None):
        return 0.
    if value < lower or value > upper:
        return -np.inf
    else:
        return 0.

def extend_theta(
    theta: np.ndarray, n_shared_pars: int, n_independent_pars: int
) -> tuple:

    idx_cut_off = n_shared_pars + 2*n_independent_pars - len(theta)
    shared_pars = np.repeat(theta[:n_shared_pars], 2)
    independent_pars = theta[n_shared_pars:idx_cut_off]

    return shared_pars, independent_pars

def prior_initialisation(
    priors: dict, preset_init_values: dict, shared_par_names: list,
    independent_par_names: list, host_galaxy_init_values: list,
    ratio_par_name: str
):

    par_names = shared_par_names + independent_par_names + [ratio_par_name]
    init_values = []
    # 3-deep conditional bleeeh
    for par in par_names:
        if par in preset_init_values.keys():
            init_par = preset_init_values[par]
        elif par in priors.keys():
            bounds = priors[par]
            if len(bounds) == 2:
                init_par = (bounds["lower"] + bounds["upper"]) / 2
            elif "lower" in bounds.keys():
                init_par = bounds["lower"] + np.random.uniform()
            elif "upper" in bounds.keys():
                init_par = bounds["upper"] - np.random.uniform()
            else:
                raise ValueError(f"{par} in prior config but bounds not defined. Check your config")
        else:
            init_par = 0.
        init_values.append(init_par)
    
    init_values = init_values + host_galaxy_init_values

    init_values = np.array(init_values)
    shared_init_pars = init_values[:len(shared_par_names)]
    independent_init_pars = np.repeat(init_values[len(shared_par_names):-1], 2)
    init_par_list = [shared_init_pars, independent_init_pars, [init_values[-1]]]
    init_pars = np.concatenate(init_par_list)

    # Check if stretch is shared and correct to account for prior
    try:
        stretch_par_name = "s"
        idx = np.repeat(independent_par_names,2).tolist().index(stretch_par_name)
        idx += len(shared_par_names)
        init_pars[idx] = init_pars[idx+1]-1.
    except Exception:
        pass

    return init_pars

def gen_pop_par_names(par_names):
    n_pars = 2 * len(par_names)
    extended_par_names = [""] * n_pars

    for i in range(0, n_pars, 2):
        extended_par_names[i] = par_names[i//2] + "_1"
        extended_par_names[i + 1] = par_names[i//2] + "_2"
    
    return extended_par_names

def theta_to_dict(
    theta: np.ndarray, shared_par_names: list, independent_par_names: list,
    ratio_par_name: str
) -> dict:

    extended_shared_par_names = gen_pop_par_names(shared_par_names)
    extended_independent_par_names = gen_pop_par_names(independent_par_names)
    missing_par_names = list(
        set([
            'gamma_Rb', 'Rb', 'sig_Rb', 'tau_Rb', 'Ebv', 'tau_Ebv'
        ]) -
        set(shared_par_names + independent_par_names)
    )
    extended_missing_par_names = gen_pop_par_names(missing_par_names)
    par_names = (
        extended_shared_par_names + extended_independent_par_names +
        extended_missing_par_names + [ratio_par_name]
    )

    n_shared_pars = len(shared_par_names)
    n_independent_pars = len(independent_par_names)

    no_pars = n_shared_pars + 2 * n_independent_pars + 1
    if len(theta) != no_pars:
        raise ValueError(
            "MCMC parameter dimensions does not match no. of shared and independent parameters."
        )

    shared_pars, independent_pars = extend_theta(theta, n_shared_pars, n_independent_pars)
    missing_pars = [NULL_VALUE] * len(extended_missing_par_names)
    par_list = [shared_pars, independent_pars, missing_pars, [theta[-1]]]
    pars = np.concatenate(par_list)
    arg_dict = {name: par for name, par in zip(par_names, pars)}

    return arg_dict

def apply_sigmoid(
    arg_dict: dict, sigmoid_cfg: dict, 
    independent_par_names: list, ratio_par_name: str
) -> dict:
    
    s1 = sigmoid(arg_dict[ratio_par_name], **sigmoid_cfg)
    s2 = sigmoid(arg_dict[ratio_par_name], scale=sigmoid_cfg['scale'], shift=1-sigmoid_cfg['shift'])

    for independent_par in independent_par_names:
        
        name1 = independent_par + "_1"
        name2 = independent_par + "_2"
        p1 = arg_dict[name1]
        p2 = arg_dict[name2]
        arg_dict[name1] = s2 * p1 + (1-s2) * p2
        arg_dict[name2] = s1 * p1 + (1-s1) * p2
    
    return arg_dict

def vectorized_apply_sigmoid(
    chains: np.ndarray, sigmoid_cfg: dict,
    shared_par_names: list
):
    
    transformed_chains = chains.copy()
    n_shared = len(shared_par_names)
    p1 = chains[:, :, n_shared:-1:2]
    p2 = chains[:, :, n_shared+1:-1:2]
    n_independent = p1.shape[-1]

    s1 = sigmoid(chains[:, :, -1], **sigmoid_cfg)
    s2 = sigmoid(chains[:, :, -1], scale=sigmoid_cfg['scale'], shift=1-sigmoid_cfg['shift'])
    s1 = np.repeat(
        sigmoid(
            chains[:, :, -1], scale=sigmoid_cfg['scale'], shift=sigmoid_cfg['shift']
        )[:, :, None],
        n_independent, -1
    )
    s2 = np.repeat(
        sigmoid(
            chains[:, :, -1], scale=sigmoid_cfg['scale'], shift=1-sigmoid_cfg['shift']
        )[:, :, None],
        n_independent, -1
    )
    
    transformed_chains[:, :, n_shared:-1:2] = s2 * p1 + (1-s2) * p2
    transformed_chains[:, :, n_shared+1:-1:2] = s1 * p1 + (1-s1) * p2

    return transformed_chains

def transformed_backend(
    current_backend, filename: str, name: str,
    sigmoid_cfg: dict, shared_par_names: list
):

    backend = em.backends.HDFBackend(filename, name=name)

    chains = vectorized_apply_sigmoid(
        current_backend.get_chain(), sigmoid_cfg,
        shared_par_names
    )
    log_prob = current_backend.get_log_prob()
    random_state = current_backend.random_state
    accepted = current_backend.accepted

    n, nwalkers, ndim = chains.shape
    backend.reset(nwalkers, ndim)
    backend.grow(n, None)

    for i in tqdm.tqdm(range(n)):
        if not i==0.:
            accepted = np.zeros_like(accepted)
        state = em.State(
            coords=chains[i], log_prob=log_prob[i],
            random_state=random_state
        )
        backend.save_step(state, accepted)

    return backend

def create_gamma_quantiles(
    lower: float, upper: float, resolution: float, cdf_limit: float
):
    vals = np.arange(lower, upper, resolution)
    quantiles = np.stack((
        vals, sp_special.gammaincinv(
            vals, cdf_limit
        )
    ))

    return quantiles

#@nb.njit
def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

@nb.njit
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

@nb.njit
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except Exception: #np.linalg.LinAlgError:
        return False

def ensure_posdef(covs: np.ndarray) -> np.ndarray:
    """Given an array of covariance matrices, check if any are not
    positive definite and, if any are found, add a small offset 
    on the diagonal to ensure positive definiteness.

    Args:
        covs (np.ndarray): Array of covaraince matrices with shape
        (N_COVS, DIM, DIM)
    Returns:
        np.ndarray: _description_
    """

    idx_neg_det = np.linalg.det(covs) <= 0
    if np.any(idx_neg_det):
        new_covs = np.empty(covs[idx_neg_det].shape)
        for i,c in enumerate(covs[idx_neg_det]):
            new_covs[i] = nearestPD(c)
        covs[idx_neg_det] = new_covs

    return covs