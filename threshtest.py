# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:40:11 2026

@author: samkr
"""
import numpy as np
from reconstructions.utils.filedirs import frequenciespkl
import pickle as pkl
import matplotlib.pyplot as plt
from reconstructions.utils import preprocess_funcs as pp

frequencies = pkl.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)

threshs = np.arange(0,51,1)
premotorNames = [[] for t in threshs]
nonpremotorNames = [[] for t in threshs]
distinct = [[] for t in threshs]
mixed = [[] for t in threshs]
VIIXII = [[] for t in threshs]
VVII = [[] for t in threshs]
XIIV = [[] for t in threshs]
XIIVIIV = [[] for t in threshs]
XII = [[] for t in threshs]
VII = [[] for t in threshs]
V = [[] for t in threshs]

for i, thresh in enumerate(threshs):
    for cell, targets in merged.iterrows():
        vii = targets['VII']
        xii = targets['XII']
        v = targets['V']
        motorTargets = [vii, xii, v]
        viixii = [vii, xii]
        vvii = [v, vii]
        xiiv = [xii, v]
        xiiviiv = [xii, vii, v]
        countTargeted = sum(1 for x in motorTargets if x > thresh)
        if countTargeted == 1:
            premotorNames[i].append(cell)
            distinct[i].append(cell)
            if vii > thresh:
                VII[i].append(cell)
            if xii > thresh:
                XII[i].append(cell)
            if v > thresh:
                V[i].append(cell)
        if countTargeted > 1:
            premotorNames[i].append(cell)
            mixed[i].append(cell)
            if sum(1 for x in xiiviiv if x > thresh) == 3:
                XIIVIIV[i].append(cell)
                continue
            if sum(1 for x in viixii if x > thresh) == 2:
                VIIXII[i].append(cell)
            if sum(1 for x in vvii if x > thresh) == 2:
                VVII[i].append(cell)
            if sum(1 for x in xiiv if x > thresh) == 2:
                XIIV[i].append(cell)
            
        if countTargeted == 0:
            nonpremotorNames.append(cell)
            
toPlot = [np.size(sublist) for sublist in premotorNames]

fig, ax = plt.subplots(dpi=300)
ax.plot(threshs, toPlot)
ax.set_xlabel('Threshold (# endpoints)')
ax.set_ylabel('# of premotor cells')

mixedoverthresh = [np.size(sub) for sub in mixed]
distoverthresh = [np.size(sub) for sub in distinct]
fig2, ax2 = plt.subplots(dpi=300)
mixline, = ax2.plot(threshs, mixedoverthresh, color='orange', label='mixed premotors')
distline, = ax2.plot(threshs, distoverthresh, color='blue', label='distinct premotors')
ax2.legend(handles=[mixline, distline])
ax2.set_xlabel('Threshold (# endpoints)')
ax2.set_ylabel('# of mixed and distinct premotor cells')
#ax2.vlines(5, 15, 40, color='black', linestyles='dashed')