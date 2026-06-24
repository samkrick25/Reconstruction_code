# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:42:59 2026
raphe cells
@author: samkr
"""
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pp
from brainrender import Scene, settings
import pickle

thresh=10

freq = pickle.load(open(frequenciespkl,'rb')).T
merged = pp.merge_regions(freq)

raphe=['RM', 'RO', 'RPA']
cells=[]
for cell, row in merged.iterrows():
    targets = row[['RM', 'RO', 'RPA']]
    for val in targets:
        if val > thresh:
            cells.append(cell)
            continue
        
