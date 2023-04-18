import hydra
import omegaconf
import os
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

import src.preprocessing as prep

NULL_VALUE = -9999.

@hydra.main(
    version_base=None, config_path="configs", config_name="datagen_config"
)
def main(cfg: omegaconf.DictConfig) -> None:

    data_path = cfg['data_cfg']['path']
    catalog = pd.read_csv(
        os.path.join(
            data_path, cfg['data_cfg']['dataset']
        ),
        sep=cfg['data_cfg']['sep']
    )

    print("\nFiltering and renaming columns...\n")
    z_and_zHD_present = 'zHD' in catalog.columns and 'z' in catalog.columns
    if z_and_zHD_present:
        print("zHD and z columns present. Using zHD as redshift column.")
        catalog = catalog.drop(columns=['z'])
    columns_to_rename = cfg['prep_cfg']['column_names']
    new_column_names = cfg['prep_cfg']['new_column_names']
    mapping = dict(zip(columns_to_rename, new_column_names))
    catalog = catalog.rename(columns=mapping)

    new_column_names += ['duplicate_uid']
    for column_name in new_column_names:
        if column_name not in catalog.columns:
            catalog[column_name] = NULL_VALUE
    
    # Flag calibration SNe
    if cfg['prep_cfg']['include_calibrator_sn']:
        print("Including calbrator SNe...\n")
        catalog = catalog.rename(
            columns={
                'IS_CALIBRATOR': 'is_calibrator',
                'CEPH_DIST': 'mu_calibrator'
            }
        )
        n_calibrator_sn = np.sum(catalog['is_calibrator'].to_numpy() == 1)
        print(f"Number of calibrator SNe: {n_calibrator_sn}\n")
    else:
        catalog['is_calibrator'] = NULL_VALUE
        catalog['mu_calibrator'] = NULL_VALUE
    new_column_names += ['is_calibrator', 'mu_calibrator']

    if cfg['prep_cfg']['use_bias_corrections']:
        print("Applying bias corrections...\n")

        apparent_mag_name = cfg['prep_cfg']['apparent_mag_column_name']
        apparent_mag_err_name = cfg['prep_cfg']['apparent_mag_err_column_name']
        apparent_mag_bias_correction_name = cfg['prep_cfg']['apparent_mag_bias_correction_column_name']
        apparent_mag_bias_correction_err_name = cfg['prep_cfg']['apparent_mag_bias_correction_err_column_name']
        
        catalog[apparent_mag_name] = catalog['mB'] - catalog[
            apparent_mag_bias_correction_name
        ]
        catalog[apparent_mag_err_name] = np.sqrt(
            catalog[apparent_mag_err_name] ** 2 +
            catalog[apparent_mag_bias_correction_err_name] ** 2
        )

    if cfg['prep_cfg']['use_redshift_cutoff']:
        print("Applying redshift cutoff...")
        redshift_column_name = 'z'
        idx_above = cfg['prep_cfg']['redshift_lower_cutoff'] < catalog[redshift_column_name].to_numpy()
        idx_below = cfg['prep_cfg']['redshift_upper_cutoff'] > catalog[redshift_column_name].to_numpy()
        idx_calibrator = catalog['is_calibrator'].to_numpy() == 1
        idx_to_keep = (idx_above & idx_below) | idx_calibrator
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog.loc[idx_to_keep].copy()
        catalog = catalog.reset_index(drop=True)
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Percentage of SNe remaining: {len(catalog) / len(idx_to_keep) * 100:.3f} %\n")
    
    if cfg['prep_cfg']['use_mb_err_cutoff']:
        print("\nApplying mB err cutoff...\n")
        alpha = cfg['prep_cfg']['alpha']
        beta = cfg['prep_cfg']['beta']
        mu_err_column_name = 'MUERR'
        idx_calibrator = catalog['is_calibrator'].to_numpy() == 1
        x1_err_column_name = 'x1Err'
        c_err_column_name = 'cErr'
        err = np.sqrt(
            (   
                catalog[mu_err_column_name].to_numpy() ** 2 +
                (alpha * catalog[x1_err_column_name].to_numpy()) ** 2 +
                (beta * catalog[c_err_column_name].to_numpy()) ** 2
            )
        )
        idx_to_keep = (err < cfg['prep_cfg']['mb_err_cutoff']) | idx_calibrator
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog.loc[idx_to_keep]
        catalog = catalog.reset_index(drop=True)
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Percentage of SNe remaining: {len(catalog) / len(idx_to_keep) * 100:.3f} %\n")

    if cfg['prep_cfg']['use_x1_cutoff']:
        print("\nApplying x1 cutoff...\n")
        x1_column_name = 'x1'
        idx_calibrator = catalog['is_calibrator'].to_numpy() == 1
        idx_to_keep = (
            np.abs(catalog[x1_column_name]).to_numpy() < cfg['prep_cfg']['x1_cutoff']
        ) | idx_calibrator
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog.loc[idx_to_keep]
        catalog = catalog.reset_index(drop=True)
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Percentage of SNe remaining: {len(catalog) / len(idx_to_keep) * 100:.3f} %\n")
    
    if cfg['prep_cfg']['use_x1_err_cutoff']:
        print("\nApplying x1 err cutoff...\n")
        x1_err_column_name = 'x1Err'
        idx_calibrator = catalog['is_calibrator'].to_numpy() == 1
        idx_to_keep = (
            np.abs(catalog[x1_err_column_name]).to_numpy() < cfg['prep_cfg']['x1_err_cutoff']
        ) | idx_calibrator
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog.loc[idx_to_keep]
        catalog = catalog.reset_index(drop=True)
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Percentage of SNe remaining: {len(catalog) / len(idx_to_keep) * 100:.3f} %\n")

    if cfg['prep_cfg']['use_color_cutoff']:
        print("\nApplying color cutoff...\n")
        color_column_name = 'c'
        idx_calibrator = catalog['is_calibrator'].to_numpy() == 1
        idx_to_keep = (
            np.abs(catalog[color_column_name]).to_numpy() < cfg['prep_cfg']['color_cutoff']
        ) | idx_calibrator
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog.loc[idx_to_keep]
        catalog = catalog.reset_index(drop=True)
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Percentage of SNe remaining: {len(catalog) / len(idx_to_keep) * 100:.3f} %\n")

    if cfg['prep_cfg']['use_fitprob_cutoff']:
        print("\nApplying fitprob cutoff...\n")
        fitprob_column_name = 'FITPROB'
        idx_calibrator = catalog['is_calibrator'].to_numpy() == 1
        idx_to_keep = (
            catalog[fitprob_column_name].to_numpy() > cfg['prep_cfg']['fitprob_cutoff']
        ) | idx_calibrator
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog.loc[idx_to_keep]
        catalog = catalog.reset_index(drop=True)
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Percentage of SNe remaining: {len(catalog) / len(idx_to_keep) * 100:.3f} %\n")

    if cfg['prep_cfg']['use_peak_date_err_cutoff']:
        print("\nApplying peak date error cutoff...\n")
        peak_date_err_column_name = 'PKMJDERR'
        idx_calibrator = catalog['is_calibrator'].to_numpy() == 1
        idx_to_keep = (
            catalog[peak_date_err_column_name].to_numpy() < cfg['prep_cfg']['peak_date_err_cutoff']
        ) | idx_calibrator
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog.loc[idx_to_keep]
        catalog = catalog.reset_index(drop=True)
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Percentage of SNe remaining: {len(catalog) / len(idx_to_keep) * 100:.3f} %\n")

    if cfg['prep_cfg']['flag_duplicate_sn']:

        print("\nIdentifying duplicate SNe...\n")

        duplicate_details = prep.identify_duplicate_sn(
            catalog,
            max_peak_date_diff=cfg['prep_cfg']['max_peak_date_diff'],
            max_angular_separation=cfg['prep_cfg']['max_angular_separation'],
        )

        number_of_sn_observations = len(catalog)
        number_of_duplicate_sn = len(duplicate_details)
        number_of_duplicate_sn_observations = 0
        dz = []
        for i, (_, sn_details) in enumerate(duplicate_details.items()):
            number_of_duplicate_sn_observations += sn_details['number_of_duplicates']
            catalog.loc[sn_details['idx_duplicate'], 'duplicate_uid'] = i
            dz.append(sn_details['dz'])

        number_of_unique_sn = number_of_sn_observations - number_of_duplicate_sn_observations + number_of_duplicate_sn

        print(f"\nNumber of SNe observations in catalog: {number_of_sn_observations}")
        print(f"Number of unique SNe in catalog: {number_of_unique_sn}")
        print(f"Number of SNe with duplicate obsevations: {number_of_duplicate_sn}")
        print(f"Total number of duplicate SNe observations: {number_of_duplicate_sn_observations}\n")
            
        print(f"\nDuplicate SNe redshift diff statistics:")
        print(f"Max dz: {np.max(dz)}")
        print(f"Min dz: {np.min(dz)}")
        print(f"Mean dz: {np.mean(dz)}")
        print(f"Median dz: {np.median(dz)}\n")

    # Morphologies
    if cfg['prep_cfg']['use_host_morphologies']:
        print("\nAdding host morphologies...\n")

        catalog['CID'] = catalog['CID'].str.lower().str.strip()
        catalog['t'] = NULL_VALUE
        catalog['t_err'] = NULL_VALUE
        new_column_names += ['t', 't_err']
        
        morphologies = pd.read_csv(
            os.path.join(
                data_path, cfg['data_cfg']['morphologies']
            ),
            sep=' ', header=0, names=['CID', 't', 't_err']
        )


        for i, row in morphologies.iterrows():
            if row['CID'] in catalog['CID'].values:
                catalog.loc[catalog['CID'] == row['CID'], 't'] = row['t']
                catalog.loc[catalog['CID'] == row['CID'], 't_err'] = row['t_err']
        
        number_of_sn_with_host_morphologies = np.sum(catalog['t'] != NULL_VALUE)

        if cfg['prep_cfg']['flag_duplicate_sn']:
            for _, value in duplicate_details.items():
                duplicate_sn_names = value['duplicate_names']
                overlap = np.isin(duplicate_sn_names, morphologies['CID'].values)
                overlap_exists = np.sum(overlap) > 0
                if overlap_exists:
                    name = duplicate_sn_names[overlap][0]
                    idx = np.isin(catalog['CID'].values, duplicate_sn_names)
                    catalog.loc[idx, 't'] = morphologies.loc[morphologies['CID'] == name, 't'].values[0]
                    catalog.loc[idx, 't_err'] = morphologies.loc[morphologies['CID'] == name, 't_err'].values[0]

        number_of_sn_with_host_morphologies = np.sum(catalog['t'] != NULL_VALUE)
        print(f"Number of SNe with host morphologies: {number_of_sn_with_host_morphologies}")
        print(f"Percentage of SNe with host morphologies: {number_of_sn_with_host_morphologies / len(catalog) * 100:.3f} %\n")

    if cfg['prep_cfg']['use_jones_properties']:
        print("\nAdding remaining host properties...\n")

        jones_catalog = pd.read_csv(
            os.path.join(
                data_path, cfg['data_cfg']['local_properties']
            ),
            sep=' '
        )

        catalog_coords = SkyCoord(
            ra=catalog['RA'].values * u.degree,
            dec=catalog['DEC'].values * u.degree
        )

        jones_coords = SkyCoord(
            ra=jones_catalog['RA'].values * u.degree,
            dec=jones_catalog['DEC'].values * u.degree
        )

        for host_property in cfg['prep_cfg']['jones_properties']:
            print(f"Adding {host_property}...\n")
            
            host_property_err = host_property + '_err'

            if not host_property in catalog.columns:
                catalog[host_property] = NULL_VALUE
                catalog[host_property_err] = NULL_VALUE
                new_column_names += [host_property, host_property_err]
            
            idx_already_set = catalog[host_property] != NULL_VALUE
            if host_property in ['global_mass', 'local_mass']:
                idx_invalid = (
                    idx_already_set & (catalog[host_property] < 5.0)
                )
                catalog.loc[idx_invalid, host_property] = NULL_VALUE
                catalog.loc[idx_invalid, host_property_err] = NULL_VALUE

            for i, jones_coord in enumerate(jones_coords):
                jones_redshift = jones_catalog.loc[i, 'z']
                idx_below_max_angular_separation = (
                    jones_coord.separation(catalog_coords) < cfg['prep_cfg']['max_angular_separation'] * u.arcsec
                )
                idx_below_angular_sep_and_not_set = (
                    idx_below_max_angular_separation & ~idx_already_set
                )
                idx_below_max_redshift_sep = (
                    np.abs(catalog['z'] - jones_redshift) < cfg['prep_cfg']['max_redshift_separation']
                )
                idx_match = (
                    idx_below_angular_sep_and_not_set & idx_below_max_redshift_sep
                )
                if np.sum(idx_match) > 0:
                    catalog.loc[idx_match, host_property] = jones_catalog.loc[i, host_property]
                    catalog.loc[idx_match, host_property_err] = jones_catalog.loc[i, host_property_err]

            idx_either_null = (
                (catalog[host_property] == NULL_VALUE) | (catalog[host_property_err] == NULL_VALUE)
            )
            catalog.loc[idx_either_null, host_property] = NULL_VALUE
            catalog.loc[idx_either_null, host_property_err] = NULL_VALUE

            idx_not_null = (
                (catalog[host_property] != NULL_VALUE) & (catalog[host_property_err] != NULL_VALUE)
            )
            idx_fractional_error_cutoff = (
                catalog[host_property_err] / np.abs(catalog[host_property]) > cfg['prep_cfg']['fractional_error_cutoff']
            ) & idx_not_null
            catalog.loc[idx_fractional_error_cutoff, host_property] = NULL_VALUE
            catalog.loc[idx_fractional_error_cutoff, host_property_err] = NULL_VALUE

            number_of_sn_with_property = np.sum(catalog[host_property] != NULL_VALUE)
            print(f"Number of SNe with {host_property}: {number_of_sn_with_property}")
            print(f"Percentage of SNe with {host_property}: {number_of_sn_with_property / len(catalog) * 100:.3f} %\n")   

    catalog = catalog[new_column_names].copy()
    save_path = os.path.join(
        cfg['prep_cfg']['output_path'],
        cfg['prep_cfg']['output_name']
    )
    catalog.to_csv(
        save_path,
        sep=cfg['prep_cfg']['output_sep'],
        index=False
    )

if __name__ == "__main__":
    main()