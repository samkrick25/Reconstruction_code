# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:17:43 2026

@author: economolab
"""

from reconstructions.utils.filedirs import parcellation_mappkl, structure_ont_info, frequenciespkl
from reconstructions.utils import preprocess_funcs, metrics_funcs
import numpy as np
import pandas as pd
#import scipy
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import pickle
import seaborn as sns

def get_list_for_region(data, regions, notcombined=None):
    "unfinished, do later"
    
    regiondf = data[regions]
    if notcombined:
        cells=[cell for cell, row in regiondf.iterrows() if np.sum(row) > 5]
    else:
        cells = [cell for cell, row in regiondf.iterrows() if np.sum(row) > 5]
    
    return cells


#os.chdir(r'C:\Users\economolab\Documents\GitHub\Reconstruction_code')
structure_to_ont = pickle.load(open(structure_ont_info, 'rb'))
parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))
frequencies_notprocessed = pickle.load(open(frequenciespkl, 'rb'))
frequencies = preprocess_funcs.preprocess(frequencies_notprocessed).T


latdf = metrics_funcs.lat_index(frequencies_notprocessed.T)

fnp = frequencies_notprocessed.T 

preprocess_funcs.write_targeted_regions_to_excel(fnp, r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\all_targets.xlsx')
# %%

freqmerged = preprocess_funcs.merge_regions(frequencies_notprocessed.T)

allpremotors = []
XII = []
VII = []
V = []
DMX = []
AMB = []

motorprojs = freqmerged[['XII', 'VII', 'V', 'DMX', 'AMB', 'VI']]
for cell, row in motorprojs.iterrows():
    totalprojs = np.sum(row)
    if totalprojs > 5:
        allpremotors.append(cell)
        
    if row['V'] > 0 and row['VII'] > 0 and row['XII'] > 0:#and np.sum(row[['AMB', 'XII', 'DMX']]) < 3:
        ...#print(cell)
#print(allpremotors)

sensory = get_list_for_region(freqmerged, regions=['PSV', 'NTS', 'PB', 'SPVI', 'SPVO', 'SPVC'])

medialreticular = get_list_for_region(freqmerged, regions=['GRN', 'MARN', 'PRNc', 'MRN'])

vestibular = get_list_for_region(freqmerged, regions=['MV', 'LAV', 'SUV', 'SPIV'])
