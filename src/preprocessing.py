import numpy as np
import pandas as pd
import astropy.units as u
import src.utils as utils

from astropy.coordinates import SkyCoord

NULL_VALUE = -9999.

# --------------- PANTHEON/SUPERCAL DATA PREPROCESSING --------------- #

def identify_duplicate_sn(
        catalog: pd.pandas.DataFrame, max_peak_date_diff: float = 10,
        max_angular_separation: float = 1, sn_id_key: str = 'CID',
        sn_redshift_key: str = 'z', sn_peak_date_key: str = 'PKMJD',
        sn_ra_key: str = 'RA', sn_dec_key: str = 'DEC'
) -> tuple:

        catalog = catalog.copy()
        sn_coordinates = SkyCoord(
            ra=catalog[sn_ra_key].to_numpy() * u.degree,
            dec=catalog[sn_dec_key].to_numpy() * u.degree,
            frame='icrs'
        )
    
        i = 0
        total_number_of_duplicate_sn = 0
        idx_of_duplicate_sn = []
        duplicate_sn_details = {}
    
        duplicate_subtracted_sn_coordinates = sn_coordinates.copy()
        duplicate_subtracted_catalog = catalog.copy()
    
        looping_condition = True
        while looping_condition:
    
            name_of_current_sn = duplicate_subtracted_catalog.iloc[i][sn_id_key]
            redshift_of_current_sn = duplicate_subtracted_catalog.iloc[i][sn_redshift_key]
            peak_date_of_current_sn = duplicate_subtracted_catalog.iloc[i][sn_peak_date_key]
    
            peak_date_diff = np.abs(
                catalog[sn_peak_date_key].to_numpy() - peak_date_of_current_sn
            )
            idx_below_max_peak_date_diff = peak_date_diff < max_peak_date_diff
    
            idx_below_max_angular_separation = duplicate_subtracted_sn_coordinates[i].separation(
                sn_coordinates
            ) < max_angular_separation * u.arcsec
    
            idx_duplicates_of_current_sn = (
                    idx_below_max_peak_date_diff & idx_below_max_angular_separation
            )
            number_of_duplicates_for_current_sn = np.count_nonzero(
                idx_duplicates_of_current_sn
            )
    
            no_duplicates_present = number_of_duplicates_for_current_sn == 1
    
            if no_duplicates_present:
                i += 1
                reached_end_of_duplicate_subtracted_catalog = (
                        i == len(duplicate_subtracted_catalog)
                )
                if reached_end_of_duplicate_subtracted_catalog:
                    looping_condition = False
                continue
    
            total_number_of_duplicate_sn += number_of_duplicates_for_current_sn

            max_abs_peak_date_diff = np.max(
                np.abs(
                    catalog[sn_peak_date_key].to_numpy()[idx_duplicates_of_current_sn] - peak_date_of_current_sn
                )
            )
            max_abs_redshift_diff = np.max(
                np.abs(
                    catalog[sn_redshift_key].to_numpy()[idx_duplicates_of_current_sn] - redshift_of_current_sn
                )
            )
            names_for_duplicates = catalog[sn_id_key].to_numpy()[idx_duplicates_of_current_sn]
    
            duplicate_sn_details[name_of_current_sn] = {
                'dz': max_abs_redshift_diff,
                'dt': max_abs_peak_date_diff,
                'number_of_duplicates': number_of_duplicates_for_current_sn,
                'duplicate_names': names_for_duplicates,
                'idx_duplicate': idx_duplicates_of_current_sn
            }
    
            idx_of_duplicate_sn.append(idx_duplicates_of_current_sn)

            idx_of_non_duplicates = ~np.any(idx_of_duplicate_sn, axis=0)
            duplicate_subtracted_catalog = catalog[idx_of_non_duplicates].copy()
            duplicate_subtracted_sn_coordinates = sn_coordinates[idx_of_non_duplicates].copy()
    
        return duplicate_sn_details

# ------------------ SN DATA PREPROCESSING ------------------ #

def init_global_data(
    data: pd.pandas.DataFrame, volumetric_rates: pd.pandas.DataFrame,
    cfg: dict, n_evaluate: int = 0
) -> tuple:

    global sn_covariances
    global sn_observables
    global gRb_quantiles
    global gEbv_quantiles
    global global_model_cfg
    global host_galaxy_observables
    global host_galaxy_covariances
    global n_unused_host_properties
    global idx_sn_to_evaluate
    global idx_duplicate_sn
    global idx_unique_sn
    global n_unique_sn
    global selection_bias_correction
    global observed_volumetric_rates
    global observed_volumetric_rate_errors
    global observed_volumetric_rate_redshifts

    # TODO: FIX THIS TO ACCOUNT FOR POTENTIAL DUPLICATES
    idx_sn_to_evaluate = data.shape[0]-n_evaluate
    global_model_cfg = cfg

    duplicate_uid_key = 'duplicate_uid'
    selection_bias_correction_key = 'bias_corr_factor'
    sn_observable_keys = ['mB', 'x1', 'c', 'z']
    sn_covariance_keys = ['x0', 'mBErr', 'x1Err', 'cErr', 'cov_x1_c', 'cov_x1_x0', 'cov_c_x0']
    sn_observables = data[sn_observable_keys].copy().to_numpy()
    sn_covariance_values = data[sn_covariance_keys].copy().to_numpy()
    data.drop(
        sn_observable_keys + sn_covariance_keys + ['CID'], axis=1, inplace=True
    )

    if duplicate_uid_key in data.columns:

        duplicate_uids = data[duplicate_uid_key].unique()
        idx_not_null = duplicate_uids != NULL_VALUE
        duplicate_uids = duplicate_uids[idx_not_null]

        idx_duplicate_sn = []
        for uid in duplicate_uids:
            idx_duplicate_sn.append(
                data['duplicate_uid'].to_numpy() == uid
            )

        idx_duplicate_sn = np.array(idx_duplicate_sn)
        idx_unique_sn = ~np.any(idx_duplicate_sn, axis=0)

        data.drop(duplicate_uid_key, axis=1, inplace=True)
    else:
        idx_duplicate_sn = []
        idx_unique_sn = np.ones((data.shape[0],), dtype=bool)

    n_unique_sn = np.count_nonzero(idx_unique_sn) + len(idx_duplicate_sn)    

    if selection_bias_correction_key in data.columns:
        selection_bias_correction = data[selection_bias_correction_key].copy().to_numpy()
        idx_null = selection_bias_correction == NULL_VALUE
        selection_bias_correction[idx_null] = 1.0
        data.drop(selection_bias_correction_key, axis=1, inplace=True)
    else:
        selection_bias_correction = np.ones((data.shape[0],))

    if cfg.get('use_volumetric_rates', False):
        observed_volumetric_rates = volumetric_rates['rate'].to_numpy()
        observed_volumetric_rate_errors = volumetric_rates['symm'].to_numpy()
        observed_volumetric_rate_redshifts = volumetric_rates['z'].to_numpy()
    else:
        observed_volumetric_rates = np.zeros((0,))
        observed_volumetric_rate_errors = np.zeros((0,))

    host_galaxy_cfg = cfg.get('host_galaxy_cfg', {})
    host_property_keys = host_galaxy_cfg.get('property_names', [])
    host_property_err_keys = [key + "_err" for key in host_property_keys]

    use_host_properties = host_galaxy_cfg.get('use_properties', False)
    can_use_host_properties = (
        set(host_property_keys) <= set(data.columns) and
        set(host_property_err_keys) <= set(data.columns)
    )
    if not use_host_properties:
        
        n_unused_host_properties = 0
        host_galaxy_observables = np.zeros((0,0))
        host_galaxy_covariance_values = np.zeros((0,0))
    
    elif can_use_host_properties:

        n_unused_host_properties = (
            len(data.columns) - len(host_property_keys) - len(host_property_err_keys)
        ) // 2
        host_galaxy_observables = data[host_property_keys].to_numpy()
        host_galaxy_observables = np.concatenate(
            (
                host_galaxy_observables,
                np.ones((host_galaxy_observables.shape[0], n_unused_host_properties)) * NULL_VALUE
            ), axis=1
        )
        host_galaxy_covariance_values = data[host_property_err_keys].to_numpy()
        host_galaxy_covariance_values = np.concatenate(
            (
                host_galaxy_covariance_values,
                np.ones((host_galaxy_covariance_values.shape[0], n_unused_host_properties)) * NULL_VALUE
            ), axis=1
        )
        host_galaxy_covariance_values = np.where(
            host_galaxy_covariance_values == NULL_VALUE,
            1 / np.sqrt(2*np.pi),
            host_galaxy_covariance_values
        )
    else:
        raise ValueError("Host galaxy properties must be provided as even columns.")

    sn_covariances, host_galaxy_covariances = build_covariance_matrix(
        sn_covariance_values, host_galaxy_covariance_values
    )

    if "prior_bounds" in cfg.keys():
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
        cov_host = np.eye(host_cov_values.shape[1]) * host_cov_values[:, None, :] ** 2
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