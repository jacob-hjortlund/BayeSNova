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
    n_independent_pars = len(independent_par_names)

    if len(theta) != n_shared_pars + 2 * n_independent_pars + 1:
        raise ValueError(
            "MCMC parameter dimensions does not match no. of shared and independent parameters."
        )

    arg_dict = {}
    for i in range(n_shared_pars):
        arg_dict[extended_shared_par_names[2 * i]] = theta[i]
        arg_dict[extended_shared_par_names[2 * i + 1]] = theta[i]

    if use_sigmoid:
        s1 = sigmoid(theta[-1], **sigmoid_cfg)
        sigmoid_cfg['shift'] = 1 - sigmoid_cfg['shift']
        s2 = sigmoid(theta[-1], **sigmoid_cfg)
        v1 = np.array([s2, s1] * n_independent_pars)
        v2 = np.array([1 - s2, 1 - s1] * n_independent_pars)
        independent_pars_1 = theta[n_shared_pars:-1:2]
        independent_pars_2 = theta[n_shared_pars+1:-1:2]
        tmp_independent_pars = np.zeros_like(v1)
        tmp_independent_pars[::2] = v1[::2] * independent_pars_1 + v2[::2] * independent_pars_2
        tmp_independent_pars[1::2] = v1[1::2] * independent_pars_1 + v2[1::2] * independent_pars_2
        for i in range(2 * n_independent_pars):
            arg_dict[extended_independent_par_names[i]] = tmp_independent_pars[i]
        # for i in range(n_independent_pars):
        #     is_odd = i % 2
        #     if not is_odd:
        #         arg_dict[extended_independent_par_names[i]] = (
        #             s2 * theta[n_shared_pars + i] + (1 - s2) * theta[n_shared_pars + i + 1]
        #         )
        #     else:
        #         arg_dict[extended_independent_par_names[i]] = (
        #             s1 * theta[n_shared_pars + i - 1] + (1 - s1) * theta[n_shared_pars + i]
        #         )
    else:
        for i in range(2 * n_independent_pars):
            arg_dict[extended_independent_par_names[i]] = theta[n_shared_pars + i]
        
    
    arg_dict[ratio_par_name] = theta[-1]

    return arg_dict

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


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
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