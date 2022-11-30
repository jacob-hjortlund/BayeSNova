import numpy as np
import src.utils as utils

# TODO: Vectorized cut functions

def build_covariance_matrix(data: np.ndarray, offset: float = 1e-323) -> np.ndarray:
    """Given a data array with shape (N,M), populate a set of 
    covariance matrices with the shape (N,3,3). If any covariance
    matrix has a zero or negative determinant, a small value is
    added along the diagonal.

    Args:
        data (np.ndarray): input array with shape (N,M)

    Returns:
        np.ndarray: (N,3,3) covariance matrix
    """

    cov = np.zeros((data.shape[0], 3, 3))

    # diagonals
    cov[:,0,0] = data[:, 1] ** 2 # omega_m^2
    cov[:,1,1] = data[:, 3] ** 2 # omega_x^2
    cov[:,2,2] = data[:, 5] ** 2 # omega_colour^2

    # upper off-diagonals
    cov[:,0,1] = data[:, 8] * (-2.5 / (np.log(10.) * data[:, 6]))
    cov[:,0,2] = data[:, 9] * (-2.5 / (np.log(10.) * data[:, 6]))
    cov[:,1,2] = data[:, 7]

    # lower off-diagonals
    cov[:,1,0] = cov[:,0,1]
    cov[:,2,0] = cov[:,0,2]
    cov[:,2,1] = cov[:,1,2]

    # add constant to covs with negative determinant
    posdef_covs = utils.ensure_posdef(cov)
    
    return posdef_covs