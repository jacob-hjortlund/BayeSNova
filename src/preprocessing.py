import numpy as np
import pandas as pd
import src.utils as utils

# TODO: Vectorized cut functions

NULL_VALUE = -9999.

def init_global_data(
    data: pd.pandas.DataFrame, cfg: dict
) -> tuple:

    global sn_covariances
    global sn_observables
    global gRb_quantiles
    global gEbv_quantiles
    global global_model_cfg

    global_model_cfg = cfg

    sn_covariances = build_covariance_matrix(data.to_numpy())
    sn_observables = data[['mB', 'x1', 'c', 'z']].to_numpy()

    gRb_quantiles = set_gamma_quantiles(cfg, 'Rb')
    gEbv_quantiles = set_gamma_quantiles(cfg, 'Ebv')


def set_gamma_quantiles(cfg: dict, par: str) -> np.ndarray:

    if cfg[par + "_integral_upper_bound"] == NULL_VALUE:
        quantiles = utils.create_gamma_quantiles(
            cfg['prior_bounds']['gamma_' + par]['lower'],
            cfg['prior_bounds']['gamma_' + par]['upper'],
            cfg['resolution_g' + par], cfg['cdf_limit_g' + par]
        )
    else:
        quantiles = None

    return quantiles

def build_covariance_matrix(data: np.ndarray) -> np.ndarray:
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