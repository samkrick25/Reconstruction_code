# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:04:58 2026
get ccf volumes
@author: samkr
"""

#import nibabel as nib
from reconstructions.utils.filedirs import allen_ccf_10um, allen_parcellationpkl, parcellation_mappkl
import numpy as np
import csv
import pickle
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

MIDLINEZ = 570
ontlevel='structure'

ccfv3 = BrainGlobeAtlas(atlas_name='allen_mouse_10um')
ann = ccfv3.annotation

# =============================================================================
# allen_ccf = nib.load(allen_ccf_10um)
# allen_ccf_data = np.asanyarray(allen_ccf.dataobj)
# =============================================================================
# =============================================================================
# allen_parcellations = pickle.load(open(allen_parcellationpkl, 'rb'))
# parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))
# parcel_ontind = parcellation_map.set_index('parcellation_term_label')
# 
# savefile = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\ccf_structure_volumes3_mm.csv'
# 
# ont2017 = np.unique(parcellation_map['parcellation_term_label'])#[304:]
# regvols = {}
# 
# for ont in ont2017:
#     vol = parcel_ontind.loc[ont]['volume_mm3']
#     reg = parcel_ontind.loc[ont]['parcellation_term_acronym']
#     if isinstance(reg, str):
#         if reg in regvols:
#             continue
#         regvols[reg] = vol
#     if isinstance(reg, pd.core.series.Series):
#         regstr = reg.values[0]
#         if regstr in regvols:
#             continue
#         volflt = sum(vol.values)
#         regvols[regstr] = volflt
# 
# with open(savefile, 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     for reg, vol in regvols.items():
#         writer.writerow([reg, vol])
# =============================================================================
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
# =============================================================================
# to_skip = ['997', '8']
# 
# ontinfo = r"D:\allen_brain_atlas\metadata\parcellation_to_parcellation_term_membership.csv"
# ontdf = pd.read_csv(ontinfo)
# odf = ontdf.set_index('parcellation_term_label')
# reg2vol = {}
# for label, row in odf.iterrows():
#     aid = label.split('-')[-1]
#     if aid in to_skip:
#         continue
#     else:
#         acr = row['parcellation_term_acronym']
#         vols = odf[odf['parcellation_term_acronym']==acr]['volume_mm3']
#         vol = np.sum(vols)
#         reg2vol[acr] = vol
# 
# reg2vol['PGRN'] = reg2vol['PGRNl'] + reg2vol['PGRNd']
# 
# savefile = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\ccf_structure_volumes4_mm.csv'
# 
# with open(savefile, 'w', newline='') as f:
#     writer = csv.writer(f, delimiter = ',')
#     for reg, vol in reg2vol.items():
#             writer.writerow([reg, vol])
# =============================================================================
