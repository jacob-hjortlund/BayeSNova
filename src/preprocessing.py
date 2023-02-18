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
    global host_galaxy_covariances

    global_model_cfg = cfg

    sn_observable_keys = ['mB', 'x1', 'c', 'z']
    sn_covariance_keys = ['x0', 'mBErr', 'x1Err', 'cErr', 'cov_x1_c', 'cov_x1_x0', 'cov_c_x0']
    sn_observables = data[sn_observable_keys].copy().to_numpy()
    sn_covariance_values = data[sn_covariance_keys].copy().to_numpy()
    data.drop(
        sn_observable_keys + sn_covariance_keys + ['CID'], axis=1, inplace=True
    )

    host_property_keys = cfg['host_galaxy_cfg']['property_names']
    host_property_err_keys = [key + "_err" for key in host_property_keys]
    use_host_properties = cfg['host_galaxy_cfg']['use_properties']
    can_use_host_properties = (
        len(host_property_keys) > 0 and
        set(host_property_keys) <= set(data.columns) and
        set(host_property_err_keys) <= set(data.columns)
    )
    if not use_host_properties:
        host_galaxy_observables = np.zeros((0,0))
        host_galaxy_covariance_values = np.zeros((0,0))
    elif can_use_host_properties:
        host_galaxy_observables = data[host_property_keys].to_numpy()
        host_galaxy_covariance_values = data[host_property_err_keys].to_numpy()
        host_galaxy_covariance_values = np.where(
            host_galaxy_covariance_values == NULL_VALUE,
            cfg['host_galaxy_cfg']['covariance_null_value'],
            host_galaxy_covariance_values
        )
    else:
        raise ValueError("Host galaxy properties must be provided as even columns.")

    sn_covariances, host_galaxy_covariances = build_covariance_matrix(
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
    a set of covariance matrices with the shape (N,3,3). Host galaxy
    property covariance matrices with shape (N,K,K) are calculated 
    given a host galaxy covariance value array with shape (N,K). SN 
    and host galaxy properties are assumed to be independent.
    Args:
        sn_cov_values (np.ndarray): SN covariance values array with shape (N,M)
        host_cov_values (np.ndarray, optional): Host galaxy covariance values array with shape (N,K). 
        Currently only handles / assumes independent properties. Defaults to None.
    Returns:
        np.ndarray: (N,3,3) covariance matrix
    """

    cov_sn = np.zeros((sn_cov_values.shape[0], 3, 3))
    if np.any(host_cov_values):
        cov_host = np.eye(host_cov_values.shape[1]) * host_cov_values[:, None, :]
        posdef_cov_host = utils.ensure_posdef(cov_host)
    else:
        posdef_cov_host = np.zeros((0,0))

    cov_sn[:,0,0] = sn_cov_values[:, 1] ** 2 # omega_m^2
    cov_sn[:,1,1] = sn_cov_values[:, 2] ** 2 # omega_x^2
    cov_sn[:,2,2] = sn_cov_values[:, 3] ** 2 # omega_colour^2

    cov_sn[:,0,1] = sn_cov_values[:, 5] * (-2.5 / (np.log(10.) * sn_cov_values[:, 0]))
    cov_sn[:,0,2] = sn_cov_values[:, 6] * (-2.5 / (np.log(10.) * sn_cov_values[:, 0]))
    cov_sn[:,1,2] = sn_cov_values[:, 4]

    cov_sn[:,1,0] = cov_sn[:,0,1]
    cov_sn[:,2,0] = cov_sn[:,0,2]
    cov_sn[:,2,1] = cov_sn[:,1,2]

    posdef_cov_sn = utils.ensure_posdef(cov_sn)
    
    return posdef_cov_sn, posdef_cov_host