import numpy as np
import pandas as pd
import astropy.units as u
import bayesnova.utils as utils

from typing import Tuple
from astropy.coordinates import SkyCoord

NULL_VALUE = -9999.
MAX_VALUE = np.finfo(np.float64).max

sn_cids = np.array([])
sn_covariances = np.array([])
sn_observables = np.array([])
sn_redshifts = np.array([])
gRb_quantiles = np.array([])
gEbv_quantiles = np.array([])
global_model_cfg = {}
host_galaxy_observables = np.array([])
host_galaxy_covariances = np.array([])
idx_host_galaxy_property_not_observed = np.array([])
n_unused_host_properties = 0
calibrator_distance_moduli = np.array([])
idx_calibrator_sn = np.array([])
idx_reordered_calibrator_sn = np.array([])
idx_duplicate_sn = np.array([])
idx_unique_sn = np.array([])
n_unique_sn = 0
selection_bias_correction = np.array([])
observed_volumetric_rates = np.array([])
observed_volumetric_rate_errors = np.array([])
observed_volumetric_rate_redshifts = np.array([])

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
    
            idx_passes_match_cuts = (
                    idx_below_max_peak_date_diff & idx_below_max_angular_separation
            )

            match_cids = catalog[sn_id_key][idx_passes_match_cuts].to_numpy()
            idx_match_cids = catalog[sn_id_key].isin(match_cids).to_numpy()
            no_of_missed_duplicates = idx_match_cids.sum() - idx_passes_match_cuts.sum()

            idx_match_cids[i] = False
            idx_duplicates_of_current_sn = idx_passes_match_cuts | idx_match_cids
            idx_missed_cids = idx_match_cids & ~idx_passes_match_cuts

            number_of_duplicates_for_current_sn = np.count_nonzero(
                idx_duplicates_of_current_sn
            ) + no_of_missed_duplicates
    
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
                'idx_duplicate': idx_duplicates_of_current_sn,
                'idx_missed_duplicates': idx_missed_cids,
                'no_of_missed_duplicates': no_of_missed_duplicates
            }

            idx_of_duplicate_sn.append(idx_duplicates_of_current_sn)

            idx_of_non_duplicates = ~np.any(idx_of_duplicate_sn, axis=0)
            duplicate_subtracted_catalog = catalog[idx_of_non_duplicates].copy()
            duplicate_subtracted_sn_coordinates = sn_coordinates[idx_of_non_duplicates].copy()
    
        return duplicate_sn_details

# ------------------ SN DATA PREPROCESSING ------------------ #

def init_global_data(
    data: pd.pandas.DataFrame, volumetric_rates: pd.pandas.DataFrame,
    cfg: dict
) -> tuple:

    global sn_cids
    global sn_covariances
    global sn_observables
    global sn_log_factors
    global sn_redshifts
    global gRb_quantiles
    global gEbv_quantiles
    global global_model_cfg
    global host_galaxy_observables
    global host_galaxy_covariances
    global host_galaxy_log_factors
    global idx_host_galaxy_property_not_observed
    global n_independent_host_properties
    global n_shared_host_properties
    global calibrator_distance_moduli
    global idx_calibrator_sn
    global idx_reordered_calibrator_sn
    global idx_duplicate_sn
    global idx_unique_sn
    global n_unique_sn
    global selection_bias_correction
    global observed_volumetric_rates
    global observed_volumetric_rate_errors
    global observed_volumetric_rate_redshifts

    calibrator_flags_available = (
        'is_calibrator' in data.columns
    )
    if calibrator_flags_available:
        idx_calibrator_sn = data['is_calibrator'].copy().to_numpy() != 1
        data.drop(['is_calibrator'], axis=1, inplace=True)
    else:
        idx_calibrator_sn = np.zeros(len(data), dtype=bool)
    
    if 'distmod_calibrator' in data.columns:
        calibrator_distance_moduli = data['distmod_calibrator'].copy().to_numpy()
        data.drop(['distmod_calibrator'], axis=1, inplace=True)
    else:
        calibrator_distance_moduli = 0.


    global_model_cfg = cfg

    duplicate_uid_key = 'duplicate_uid'
    selection_bias_correction_key = 'bias_corr_factor'
    sn_redshift_key = 'z'
    sn_observable_keys = ['mB', 'x1', 'c']
    sn_covariance_keys = ['x0', 'mBErr', 'x1Err', 'cErr', 'cov_x1_c', 'cov_x1_x0', 'cov_c_x0']
    sn_observables = data[sn_observable_keys].copy().to_numpy()
    sn_redshifts = data[sn_redshift_key].copy().to_numpy().squeeze()
    sn_covariance_values = data[sn_covariance_keys].copy().to_numpy()
    sn_cids = data['CID'].copy().to_numpy()
    to_drop = sn_observable_keys + sn_covariance_keys + ['z','CID']
    if 'SurveyID' in data.columns:
        to_drop.append('SurveyID')
    data.drop(
        to_drop, axis=1, inplace=True
    )

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

    sn_covariances = build_covariance_matrix(
        sn_covariance_values
    )

    any_duplicates_present = False
    idx_duplicate_sn = []
    idx_unique_sn = np.ones((data.shape[0],), dtype=bool)
    idx_reordered_calibrator_sn = idx_calibrator_sn
    duplicate_uid_available = duplicate_uid_key in data.columns

    if duplicate_uid_available:
        duplicate_uids = data[duplicate_uid_key].copy().unique()
        idx_not_null = duplicate_uids != NULL_VALUE
        any_duplicates_present = np.any(idx_not_null)

    if any_duplicates_present:

        duplicate_uids = duplicate_uids[idx_not_null]

        idx_duplicate_sn = []
        for uid in duplicate_uids:
            idx_duplicate_sn.append(
                data['duplicate_uid'].copy().to_numpy() == uid
            )

        idx_duplicate_sn = np.array(idx_duplicate_sn)
        idx_unique_sn = ~np.any(idx_duplicate_sn, axis=0)

        idx_unique_calibrator_sn, idx_duplicate_calibrator_sn = reorder_duplicates(
            idx_calibrator_sn, idx_unique_sn, idx_duplicate_sn
        )
        idx_duplicate_calibrator_sn = np.array(
            [
                np.any(tmp_idx) for tmp_idx in idx_duplicate_calibrator_sn
            ]
        )
        idx_calibrator_sn = np.concatenate(
            (idx_unique_calibrator_sn, idx_duplicate_calibrator_sn)
        )

    sn_cids = np.concatenate(
        reorder_duplicates(
            sn_cids, idx_unique_sn,
            idx_duplicate_sn
        )
    )

    data.drop(duplicate_uid_key, axis=1, inplace=True, errors='ignore')
    n_unique_sn = np.count_nonzero(idx_unique_sn) + len(idx_duplicate_sn)

    sn_redshifts, _, _ = reduce_duplicates(
        sn_redshifts[:, None], np.ones((len(data), 1,1)),
        idx_unique_sn, idx_duplicate_sn
    )
    sn_redshifts = sn_redshifts.squeeze()
    sn_observables, sn_covariances, sn_log_factors = reduce_duplicates(
        sn_observables, sn_covariances, idx_unique_sn, idx_duplicate_sn
    )

    if calibrator_flags_available:
        tmp_cov = np.ones((len(data), 1, 1))
        idx_not_calibrator = calibrator_distance_moduli == NULL_VALUE
        tmp_cov[idx_not_calibrator,:,:] = NULL_VALUE
        calibrator_distance_moduli, _, _ = reduce_duplicates(
            calibrator_distance_moduli[:, None], tmp_cov,
            idx_unique_sn, idx_duplicate_sn
        )
        calibrator_distance_moduli = calibrator_distance_moduli[idx_calibrator_sn].squeeze()

    host_galaxy_cfg = cfg.get('host_galaxy_cfg', {})
    independent_host_property_keys = host_galaxy_cfg.get('independent_property_names', [])
    independent_host_property_err_keys = [
        key + "_err" for key in independent_host_property_keys
    ]
    shared_host_property_keys = host_galaxy_cfg.get('shared_property_names', [])
    shared_host_property_err_keys = [
        key + "_err" for key in shared_host_property_keys
    ]
    host_property_keys = (
        shared_host_property_keys + independent_host_property_keys
    )
    host_property_err_keys = (
        shared_host_property_err_keys + independent_host_property_err_keys
    )

    use_host_properties = host_galaxy_cfg.get('use_properties', False)
    can_use_host_properties = (
        set(host_property_keys) <= set(data.columns) and
        set(host_property_err_keys) <= set(data.columns)
    )
    
    if not use_host_properties:
        
        n_independent_host_properties = 0
        n_shared_host_properties = 0
        host_galaxy_observables = np.zeros((0,0))
        host_galaxy_covariance_values = np.zeros((0,0))
        host_galaxy_log_factors = np.zeros(n_unique_sn)
        idx_host_galaxy_property_not_observed = np.ones(
                (n_unique_sn, 1), dtype=bool
        )

    elif can_use_host_properties and use_host_properties:
        
        n_independent_host_properties = len(independent_host_property_keys)
        n_shared_host_properties = len(shared_host_property_keys)

        host_galaxy_observables = data[host_property_keys].to_numpy()
        host_galaxy_covariance_values = data[host_property_err_keys].to_numpy()
        host_galaxy_covariance_values = np.where(
            host_galaxy_covariance_values != NULL_VALUE,
            host_galaxy_covariance_values**2 + host_galaxy_cfg['error_floor']**2,
            MAX_VALUE,
        )
        host_galaxy_covariances = (
            np.eye(host_galaxy_covariance_values.shape[1]) *
            host_galaxy_covariance_values[:, None, :]
        )


        host_galaxy_observables, host_galaxy_covariances, host_galaxy_log_factors = reduce_duplicates(
            host_galaxy_observables, host_galaxy_covariances,
            idx_unique_sn, idx_duplicate_sn
        )
        
        host_galaxy_covariances = np.array(
            [np.diag(tmp_cov) for tmp_cov in host_galaxy_covariances]
        )

        idx_host_galaxy_property_not_observed = host_galaxy_observables == NULL_VALUE
    else:
        raise ValueError("Host galaxy properties must be provided as even columns.")


    if "prior_bounds" in cfg.keys():
        gRb_quantiles = set_gamma_quantiles(cfg, 'Rb')
        gEbv_quantiles = set_gamma_quantiles(cfg, 'Ebv')

def set_gamma_quantiles(cfg: dict, par: str) -> np.ndarray:

    if cfg.get(par + "_integral_upper_bound", None) == NULL_VALUE:
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

    singular_values = np.linalg.svd(posdef_cov_sn)[1]
    max_singular_value = np.max(singular_values, axis=1)
    min_singular_value = np.min(singular_values, axis=1)
    idx_singular_value_ratio = min_singular_value / max_singular_value < 1e-5    
    idx_condition = -np.log10(np.linalg.cond(posdef_cov_sn)) < np.log10(np.finfo(np.float64).eps)
    idx_almost_singular = np.logical_or(idx_singular_value_ratio, idx_condition)
    posdef_cov_sn[idx_almost_singular] = posdef_cov_sn[idx_almost_singular] + np.diag(np.ones(3)*1e-10)

    return posdef_cov_sn

def reorder_duplicates(
    sn_array: np.ndarray, idx_unique_sn: np.ndarray,
    idx_duplicate_sn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder duplicate SN arrays to match the order of the unique SN array.

    Args:
        sn_array (np.ndarray): Array of SN properties with shape (N,....)
        idx_unique_sn (np.ndarray): Array of boolean indices of unique SN with shape (N,)
        idx_duplicate_sn (np.ndarray): Array of boolean indices of duplicate SN with shape (M, N)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Reordered unique and duplicate SN arrays. Unique array has
        shape (K, ...) and duplicate array is an object array with shape (M, (L_i, ...))
    """
    
    duplicate_sn_array = []
    for idx in idx_duplicate_sn:
        duplicate_sn_array.append(sn_array[idx])
    duplicate_sn_array = np.array(duplicate_sn_array, dtype=object)

    unique_sn_array = sn_array[idx_unique_sn].copy()

    return unique_sn_array, duplicate_sn_array

def reduced_observables_and_covariances(
    duplicate_covariances: np.ndarray, duplicate_observables: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce the number of observables and covariances by combining
    duplicates.

    Args:
        duplicate_covariances (np.ndarray): Array of duplicate covariance matrices with shape (N,3,3)
        duplicate_observables (np.ndarray): Array of duplicate observable arrays with shape (N,3)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Reduced covariance matrices and observables
    """

    n_duplicates = len(duplicate_covariances)
    if n_duplicates == 0:
        n_dimensions = 0
    else:
        n_dimensions = duplicate_covariances[0].shape[-1]

    reduced_observables = np.zeros((n_duplicates, n_dimensions))
    reduced_covariances = np.zeros((n_duplicates, n_dimensions, n_dimensions))
    reduced_log_factors = np.zeros(n_duplicates)

    for i in range(n_duplicates):

        duplicate_obs = duplicate_observables[i][:, :, None].astype(np.float64)
        duplicate_covs = duplicate_covariances[i].astype(np.float64)
        duplicate_covs = np.where(
            duplicate_covs == NULL_VALUE, MAX_VALUE, duplicate_covs
        )
        max_value_in_cov = np.any(duplicate_covs == MAX_VALUE)
        if max_value_in_cov:
            inv_function = np.linalg.inv
        else:
            inv_function = np.linalg.pinv

        cov_i_inv = inv_function(duplicate_covs)
        reduced_cov = inv_function(
            np.sum(
                cov_i_inv, axis=0
            )
        )

        reduced_obs = np.matmul(
            reduced_cov, np.sum(
                np.matmul(cov_i_inv, duplicate_obs), axis=0
            )
        ).squeeze()

        log_factor_first_term = np.sum(
            n_dimensions * np.log(2 * np.pi) * np.ones(duplicate_obs.shape[0]) +
            np.linalg.slogdet(duplicate_covs)[1] +
            np.matmul(
                duplicate_obs.transpose(0, 2, 1),
                np.matmul(cov_i_inv, duplicate_obs)
            ).squeeze()
        )
        log_factor_second_term = (
            n_dimensions * np.log(2 * np.pi) +
            np.linalg.slogdet(reduced_cov)[1] +
            np.matmul(
                np.atleast_1d(reduced_obs).T, np.matmul(
                    inv_function(reduced_cov), np.atleast_1d(reduced_obs)
                )
            )
        )

        reduced_log_factors[i] = -0.5 * (
            log_factor_first_term - log_factor_second_term
        )
        reduced_observables[i] = reduced_obs
        reduced_covariances[i] = reduced_cov
    
    return reduced_observables, reduced_covariances, reduced_log_factors

def reduce_duplicates(
    observables: np.ndarray, covariances: np.ndarray,
    idx_unique_sn: np.ndarray, idx_duplicate_sn: np.ndarray,
):
    
    not_available = len(observables) == 0
    if not_available:
        return observables, covariances

    unique_observables, duplicate_observables = reorder_duplicates(
        observables, idx_unique_sn, idx_duplicate_sn
    )
    unique_covariances, duplicate_covariances = reorder_duplicates(
        covariances, idx_unique_sn, idx_duplicate_sn
    )

    n_unique_sn = len(unique_observables)
    n_duplicate_observables = len(duplicate_observables)
    log_factors = np.zeros(n_unique_sn)
    if n_duplicate_observables == 0:
        return unique_observables, unique_covariances, log_factors

    else:
        reduced_observables, reduced_covariances, duplicate_log_factors = reduced_observables_and_covariances(
            duplicate_covariances, duplicate_observables
        )

        reduced_observables = np.concatenate(
            (unique_observables, reduced_observables), axis=0
        )

        reduced_covariances = np.concatenate(
            (unique_covariances, reduced_covariances), axis=0
        )

        log_factors = np.concatenate(
            (log_factors, duplicate_log_factors), axis=0
        )

        return reduced_observables, reduced_covariances, log_factors




