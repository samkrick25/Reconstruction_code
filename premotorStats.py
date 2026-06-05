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
from reconstructions.utils import preprocess_funcs as pp

def get_projection_vals_from_reference(df, targets, refList, mult):
    match mult:
        case False:
            target = targets[0]
            return [val[target] for cell, val in df.iterrows() if cell in refList]
        case True:
            toRet = []
            match len(targets):
                case 2:
                    one, two = (targets[0], targets[1])
                    for cell, val in df.iterrows():
                        if cell in refList:
                            s = val[one] + val[two]
                            toRet.append(s)
                case 3:
                    one, two, three = (targets[0], targets[1], targets[2])
                    for cell, val in df.iterrows():
                        if cell in refList:
                            s = val[one] + val[two] + val[three]
                            toRet.append(s)
            return toRet

frequencies = pkl.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)

thresh = 0
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
    countTargeted = sum(1 for x in motorTargets if x > thresh)
    if countTargeted == 1:
        distinct.append(cell)
        if vii > thresh:
            VII.append(cell)
        if xii > thresh:
            XII.append(cell)
        if v > thresh:
            V.append(cell)
    if countTargeted > 1:
        mixed.append(cell)
        if sum(1 for x in xiiviiv if x > thresh) == 3:
            XIIVIIV.append(cell)
            continue
        if sum(1 for x in viixii if x > thresh) == 2:
            VIIXII.append(cell)
        if sum(1 for x in vvii if x > thresh) == 2:
            VVII.append(cell)
        if sum(1 for x in xiiv if x > thresh) == 2:
            XIIV.append(cell)
        
    if countTargeted == 0:
        nonpremotorNames.append(cell)

mult = [VIIXII, VVII, XIIV, XIIVIIV]
multflat = [item for sub in mult for item in sub]

alabels = ['1 motor nucleus', 'Multiple motor nuclei']
dlabels = ['XII', 'V', 'VII', 'Combination of XII/V/VII']
dlabels2 = ['XII', 'V', 'VII', 'XII/V', 'XII/VII', 'V/VII', 'XII/V/VII']
acounts = [np.size(distinct), np.size(mixed)]
dcounts = [np.size(XII), np.size(V), np.size(VII), np.size(mixed)]
dcounts2 = dcounts[0:3]+[np.size(XIIV), np.size(VIIXII), np.size(VVII), np.size(XIIVIIV)]

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

fig3, ax3 = plt.subplots(dpi=300)
ax3.pie(dcounts2, labels=dlabels2, colors=colors, autopct='%1.1f%%')
fig3.suptitle('Unique and Mixed Motor Nucleus Targeting of IRN/PARN cells')

#preprocess to normalize for cell size, then plot histograms of both normalized
#and non normalized endpoints for each population of motor nuc projection pattern
pp_freq = pp.preprocess(merged, pct=False, log1p=False)

XIIpct = get_projection_vals_from_reference(pp_freq, ['XII'], XII, mult=False)
VIIpct = get_projection_vals_from_reference(pp_freq, ['VII'], VII, mult=False)
Vpct = get_projection_vals_from_reference(pp_freq, ['V'], V, mult=False)
XIIVIIpct = get_projection_vals_from_reference(pp_freq, ['XII', 'VII'], VIIXII, mult=True)
XIIVpct = get_projection_vals_from_reference(pp_freq, ['XII', 'V'], XIIV, mult=True)
VVIIpct = get_projection_vals_from_reference(pp_freq, ['V', 'VII'], VVII, mult=True)
XIIVIIVpct = get_projection_vals_from_reference(pp_freq, ['XII', 'VII', 'V'], XIIVIIV, mult=True)

uniqfig, [XIIax, VIIax, Vax] = plt.subplots(3,1, figsize=(8,8), dpi=300)
XIIax.hist(XIIpct, bins=50)
uniqfig.suptitle('# endpoints in each MN for distinct premotors')
uniqfig.supxlabel('# endpoints')
uniqfig.supylabel('# cells')
XIIax.set_title('XII')
VIIax.hist(VIIpct, bins=50)
VIIax.set_title('VII')
Vax.hist(Vpct, bins=50)
Vax.set_title('V')
uniqfig.tight_layout()

mixfig, [XIIVIIax, XIIVax, VVIIax, XIIVIIVax] = plt.subplots(4,1,figsize=(8,12), dpi=300)
mixfig.suptitle('mean # endpoints in each MN for mixed premotors')
mixfig.supxlabel('mean # endpoints')
mixfig.supylabel('# cells')
XIIVIIax.hist([val/2 for val in XIIVIIpct], bins=50)
XIIVIIax.set_title('XII/VII')
XIIVax.hist([val/2 for val in XIIVpct], bins=50)
XIIVax.set_title("XII/V")
VVIIax.hist([val/2 for val in VVIIpct], bins=50)
VVIIax.set_title("V/VII")
XIIVIIVax.hist([val/3 for val in XIIVIIVpct], bins=50)
XIIVIIVax.set_title('XII/VII/V')
mixfig.tight_layout()
# =============================================================================
# distsumfig, dax = plt.subplots(dpi=300)
# distsumfig.suptitle(f'# premotor endpoints for distinct premotors n={np.size(XIIpct+VIIpct+Vpct)}')
# dc, dv = np.histogram(np.array(XIIpct+VIIpct+Vpct))
# dax.plot(dv, dc)
# dax.set_xlabel('# endpoints')
# dax.set_ylabel('# cells')
# 
# mixsumfig, iax = plt.subplots(dpi=300)
# mixsumfig.suptitle(f'# premotor endpoints for mixed premotors n={np.size(XIIVIIpct+VVIIpct+XIIVpct+XIIVIIVpct)}')
# mc, mv = np.histogram(np.array(XIIVIIpct+VVIIpct+XIIVpct+XIIVIIVpct))
# iax.plot(mv, mc)
# iax.set_xlabel('# endpoints')
# iax.set_ylabel('# cells')
# =============================================================================

