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
    global host_galaxy_observables

    global_model_cfg = cfg

    sn_observable_keys = ['mB', 'x1', 'c', 'z']
    sn_covariance_keys = ['x0', 'mBErr', 'x1Err', 'cErr', 'cov_x1_c', 'cov_x1_x0', 'cov_c_x0']
    sn_observables = data[sn_observable_keys].copy().to_numpy()
    sn_covariance_values = data[sn_covariance_keys].copy().to_numpy()
    data.drop(
        sn_observable_keys + sn_covariance_keys + ['CID'], axis=1, inplace=True
    )

    if data.shape[1] == 0:
        host_galaxy_observables = None
        host_galaxy_covariance_values = None
    elif (data.shape[1] % 2) == 0:
        host_galaxy_observables = data.to_numpy()[:, ::2]
        host_galaxy_covariance_values = data.to_numpy()[:, 1::2]
    else:
        raise ValueError("Host galaxy properties must be provided as even columns.")

    sn_covariances = build_covariance_matrix(
        sn_covariance_values, host_galaxy_covariance_values
    )
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

def build_covariance_matrix(
    sn_cov_values: np.ndarray,
    host_cov_values: np.ndarray = None,
) -> np.ndarray:
    """Given a SN covariance value array with shape (N,M), populate
    a set of covariance matrices with the shape (N,3,3). If any host
    galaxy properties are provided (with shape (N,K)), they are added
    to the covariance matrix. SN and host galaxy properties are assumed
    to be independent. If any covariance matrix has a zero or negative
    determinant, a small value is added along the diagonal.

    Args:
        sn_cov_values (np.ndarray): SN covariance values array with shape (N,M)
        host_cov_values (np.ndarray, optional): Host galaxy covariance values array with shape (N,K). 
        Currently only handles / assumes independent properties. Defaults to None.
    Returns:
        np.ndarray: (N,3,3) covariance matrix
    """

    shape = 3
    if np.any(host_cov_values):
        shape += host_cov_values.shape[1]
        cov = np.zeros((sn_cov_values.shape[0], shape, shape))
        diag_host_covs = np.eye(host_cov_values.shape[1]) * host_cov_values[:, None, :]
        cov[:,3:,3:] = diag_host_covs
    else:
        cov = np.zeros((sn_cov_values.shape[0], 3, 3))

    # diagonals
    cov[:,0,0] = sn_cov_values[:, 1] ** 2 # omega_m^2
    cov[:,1,1] = sn_cov_values[:, 2] ** 2 # omega_x^2
    cov[:,2,2] = sn_cov_values[:, 3] ** 2 # omega_colour^2

    # upper off-diagonals
    cov[:,0,1] = sn_cov_values[:, 5] * (-2.5 / (np.log(10.) * sn_cov_values[:, 0]))
    cov[:,0,2] = sn_cov_values[:, 6] * (-2.5 / (np.log(10.) * sn_cov_values[:, 0]))
    cov[:,1,2] = sn_cov_values[:, 4]

    # lower off-diagonals
    cov[:,1,0] = cov[:,0,1]
    cov[:,2,0] = cov[:,0,2]
    cov[:,2,1] = cov[:,1,2]

    # add constant to covs with negative determinant
    posdef_covs = utils.ensure_posdef(cov)
    
    return posdef_covs