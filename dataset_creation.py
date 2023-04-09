import hydra
import omegaconf
import os
import numpy as np
import pandas as pd

import src.preprocessing as prep

NULLL_VALUE = -9999.

@hydra.main(
    version_base=None, config_path="configs", config_name="datagen_config"
)
def main(cfg: omegaconf.DictConfig) -> None:

    catalog = pd.read_csv(
        cfg['data_cfg']['path'], sep=cfg['data_cfg']['sep']
    )
    catalog['duplicate_uid'] = NULLL_VALUE

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

    # TODO: FILTER AND RENAME COLUMNS
    print("\nFiltering and renaming columns...\n")
    columns_to_rename = cfg['prep_cfg']['column_names']
    new_column_names = cfg['prep_cfg']['new_column_names']
    mapping = dict(zip(columns_to_rename, new_column_names))
    catalog = catalog.rename(columns=mapping)

    new_column_names += ['duplicate_uid']

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
        idx_to_keep = cfg['prep_cfg']['redshift_cutoff'] <= catalog[redshift_column_name]
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog[idx_to_keep]
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Fraction of SNe remaining: {len(catalog) / len(idx_to_keep):.3f}\n")
    
    if cfg['prep_cfg']['use_x1_cutoff']:
        print("\nApplying x1 cutoff...\n")
        x1_column_name = 'x1'
        idx_to_keep = np.abs(catalog[x1_column_name]) < cfg['prep_cfg']['x1_cutoff']
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog[idx_to_keep]
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Fraction of SNe remaining: {len(catalog) / len(idx_to_keep):.3f}\n")
    
    if cfg['prep_cfg']['use_color_cutoff']:
        print("\nApplying color cutoff...\n")
        color_column_name = 'c'
        idx_to_keep = np.abs(catalog[color_column_name]) < cfg['prep_cfg']['color_cutoff']
        n_sn_filtered = len(catalog) - np.sum(idx_to_keep)
        catalog = catalog[idx_to_keep]
        print(f"Number of SNe filtered: {n_sn_filtered}")
        print(f"Number of SNe remaining: {len(catalog)}")
        print(f"Fraction of SNe remaining: {len(catalog) / len(idx_to_keep):.3f}\n")


    print(1)
    # TODO: ADD ADDITIONAL HOST PROPERTY COLUMNS

    # TODO: SYMMETRIZE ERRORS

if __name__ == "__main__":
    main()