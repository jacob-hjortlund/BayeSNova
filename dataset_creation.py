import pandas as pd
import numpy as np

import src.preprocessing as prep

NULLL_VALUE = -9999.

catalog = pd.read_csv("/home/jacob/Uni/Msc/Thesis/data/Pantheon+SH0ES.dat", sep=' ')
catalog['duplicate_uid'] = NULLL_VALUE

duplicate_details = prep.identify_duplicate_sn(
    catalog, max_peak_date_diff=10, max_angular_separation=1.
)

number_of_sn_observations = len(catalog)
number_of_duplicate_sn = len(duplicate_details)
number_of_duplicate_sn_observations = 0
for sn_name, sn_details in duplicate_details.items():
    number_of_duplicate_sn_observations += sn_details['number_of_duplicates']

number_of_unique_sn = number_of_sn_observations - number_of_duplicate_sn_observations + number_of_duplicate_sn

print(f"\nNumber of SNe observations in catalog: {number_of_sn_observations}")
print(f"Number of unique SNe in catalog: {number_of_unique_sn}")
print(f"Number of SNe with duplicate obsevations: {number_of_duplicate_sn}")
print(f"Total number of duplicate SNe observations: {number_of_duplicate_sn_observations}\n")

dz = []
for i, (sn_name, sn_details) in enumerate(duplicate_details.items()):
    catalog.loc[sn_details['idx_duplicate'], 'duplicate_uid'] = i
    dz.append(sn_details['dz'])

print(f"Max dz: {np.max(dz)}")
print(f"Min dz: {np.min(dz)}")
print(f"Mean dz: {np.mean(dz)}")
print(f"Median dz: {np.median(dz)}")

