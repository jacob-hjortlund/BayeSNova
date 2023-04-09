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

    # TODO: ADD ADDITIONAL HOST PROPERTY COLUMNS

    # TODO: SYMMETRIZE ERRORS

if __name__ == "__main__":
    main()