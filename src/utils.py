import numpy as np

def sigmoid(value: float, shift: float = 0., scale: float = 1.):

    denom = 1. + np.exp(- scale * ( value - shift ))

    return 1. / denom

def uniform(value: float, lower: float = -np.inf, upper: float = np.inf):
    if value < lower or value > upper:
        return -np.inf
    else:
        return 0.

def gen_pop_par_names(par_names):
    n_pars = 2 * len(par_names)
    extended_par_names = [""] * n_pars

    for i in range(0, n_pars, 2):
        extended_par_names[i] = par_names[i//2] + "_1"
        extended_par_names[i + 1] = par_names[i//2] + "_2"
    
    return extended_par_names


def theta_to_dict(
    theta: np.ndarray, shared_par_names: list, independent_par_names: list,
    ratio_par_name: str, use_sigmoid: bool, sigmoid_cfg: dict
) -> dict:

    extended_shared_par_names = gen_pop_par_names(shared_par_names)
    extended_independent_par_names = gen_pop_par_names(independent_par_names)

    n_shared_pars = len(shared_par_names)
    n_independent_pars = len(extended_independent_par_names)

    if len(theta) != n_shared_pars + n_independent_pars + 1:
        raise ValueError(
            "MCMC parameter dimensions does not match no. of shared and independent parameters."
        )

    arg_dict = {}
    for i in range(n_shared_pars):
        arg_dict[extended_shared_par_names[i]] = theta[i]
        arg_dict[extended_shared_par_names[i + 1]] = theta[i]

    if use_sigmoid:
        for i in range(n_independent_pars):
            is_even = i % 2
            s1 = sigmoid(theta[-1], **sigmoid_cfg)
            sigmoid_cfg['shift'] = 1 - sigmoid_cfg['shift']
            s2 = sigmoid_cfg(theta[-1], **sigmoid_cfg)
            if is_even:
                arg_dict[extended_independent_par_names[i]] = (
                    (1 - s2) * theta[n_shared_pars + i] + s2 * theta[n_shared_pars + i + 1]
                )
            else:
                arg_dict[extended_independent_par_names[i]] = (
                    (1 - s1) * theta[n_shared_pars + i] + s1 * theta[n_shared_pars + i - 1]
                )
    else:
        for i in range(n_independent_pars):
            arg_dict[extended_independent_par_names[i]] = theta[n_shared_pars + i]
        
    
    arg_dict[ratio_par_name] = theta[-1]

    return arg_dict

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
    covs[idx_neg_det] += np.diag(
        np.ones(covs.shape[1]) * np.finfo("float64").eps
    )

    return covs