import numpy as np

def ensure_posdef(
    covs: np.ndarray, offset: float = 1e-323
) -> np.ndarray:
    """Given an array of covariance matrices, check if any are not
    positive definite and, if any are found, add a small offset 
    on the diagonal to ensure positive definiteness.

    Args:
        covs (np.ndarray): Array of covaraince matrices with shape
        (N_COVS, DIM, DIM)
        offset (float, optional): Diagonal offset. Defaults to 1e-323.

    Returns:
        np.ndarray: _description_
    """

    idx_neg_det = np.linalg.det(covs) <= 0
    covs[idx_neg_det] += np.diag(
        np.ones(covs.shape[1]) * offset
    )

    return covs