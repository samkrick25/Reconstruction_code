# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:50:06 2026
premotor stats
@author: samkr
"""

import pandas as pd
import numpy as np
from reconstructions.utils.filedirs import frequenciespkl
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
from reconstructions.utils import preprocess_funcs as pf

frequencies = pkl.load(open(frequenciespkl, 'rb')).T
merged = pf.merge_regions(frequencies)

premotorNames = []
nonpremotorNames = []
distinct = []
mixed = []
VIIXII = []
VVII = []
XIIV = []
XIIVIIV = []
XII = []
VII = []
V = []

for cell, targets in merged.iterrows():
    vii = targets['VII']
    xii = targets['XII']
    v = targets['V']
    motorTargets = [vii, xii, v]
    viixii = [vii, xii]
    vvii = [v, vii]
    xiiv = [xii, v]
    xiiviiv = [xii, vii, v]
    countTargeted = sum(1 for x in motorTargets if x != 0)
    if countTargeted == 1:
        distinct.append(cell)
        if vii > 0:
            VII.append(cell)
        if xii > 0:
            XII.append(cell)
        if v > 0:
            V.append(cell)
    if countTargeted > 1:
        mixed.append(cell)
        if sum(1 for x in xiiviiv if x != 0) == 3:
            XIIVIIV.append(cell)
            continue
        if sum(1 for x in viixii if x != 0) == 2:
            VIIXII.append(cell)
        if sum(1 for x in vvii if x != 0) == 2:
            VVII.append(cell)
        if sum(1 for x in xiiv if x != 0) == 2:
            XIIV.append(cell)
        
    if countTargeted == 0:
        nonpremotorNames.append(cell)

mult = [VIIXII, VVII, XIIV, XIIVIIV]
multflat = [item for sub in mult for item in sub]

alabels = ['1 motor nucleus', 'Multiple motor nuclei']
dlabels = ['XII', 'V', 'VII', 'Combination of XII/V/VII']
acounts = [np.size(distinct), np.size(mixed)]
dcounts = [np.size(XII), np.size(V), np.size(VII), np.size(mixed)]

colors = mpl.colormaps['Set1'].colors
ctouse = colors[1:5]

# =============================================================================
# fig, [aax, dax] = plt.subplots(1,2, figsize=(10,8), dpi=300)
# 
# aax.pie(acounts, labels=alabels, colors='Dark2')
# =============================================================================
fig, ax = plt.subplots(dpi=300)
ax.pie(dcounts, labels=dlabels, colors=colors[1:6], autopct='%1.1f%%')
fig.suptitle('Mixed vs. Distinct Premotor Targets')

fig2, ax2 = plt.subplots(dpi=300)
ax2.pie([np.size(nonpremotorNames), sum(dcounts)], labels=['Non-premotor', 'Premotor'], autopct='%1.1f%%')
fig2.suptitle('Premotor vs. Non-premotor')
