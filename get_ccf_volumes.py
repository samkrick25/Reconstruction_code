# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:04:58 2026
get ccf volumes
@author: samkr
"""

import nibabel as nib
from reconstructions.utils.filedirs import allen_ccf_10um, allen_parcellationpkl, parcellation_mappkl
import numpy as np
import csv
import pickle
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

MIDLINEZ = 570
ontlevel='structure'

# =============================================================================
# allen_ccf = nib.load(allen_ccf_10um)
# allen_ccf_data = np.asanyarray(allen_ccf.dataobj)
# =============================================================================
allen_parcellations = pickle.load(open(allen_parcellationpkl, 'rb'))
parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))
parcel_ontind = parcellation_map.set_index('parcellation_term_label')

savefile = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\ccf_structure_volumes2_mm.csv'

ont2017 = np.unique(parcellation_map['parcellation_term_label'])[304:]
regvols = {}

for ont in ont2017:
    vol = parcel_ontind.loc[ont]['volume_mm3']
    reg = parcel_ontind.loc[ont]['parcellation_term_acronym']
    if isinstance(reg, str):
        regvols[reg] = vol
    if isinstance(reg, pd.core.series.Series):
        regstr = reg.values[0]
        volflt = vol.values[0]
        regvols[regstr] = volflt

with open(savefile, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for reg, vol in regvols.items():
        writer.writerow([reg, vol])
# =============================================================================
# ccf_onehem = allen_ccf_data[:,:,:570]
# 
# regcounts = defaultdict(int)
# ids, counts = np.unique(ccf_onehem, return_counts=True)
# for id, count in zip(ids, counts):
#     #get mm^3 of each region
#     regcounts[id] = (count*10)#*1e-9
# =============================================================================
# =============================================================================
# for x in tqdm(ccf_onehem, desc='scanning AP'):
#     for y in tqdm(x, desc='scanning DV'):
#         ids, counts = np.unique
# =============================================================================

# =============================================================================
# with open(savefile, 'w', newline='') as f:
#     writer = csv.writer(f, delimiter = ',')
#     for reg, vol in regcounts.items():
#             parcels = parcellation_map.loc[allen_parcellations.loc[reg]['label']]
#             region = get_allen_region(ontlevel, parcels)
#             writer.writerow([region, vol])
# =============================================================================
